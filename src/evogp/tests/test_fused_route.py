"""Regression net for the shipped ``EVOGP_EVAL_MODE=fused`` (Triton) route.

``src/evogp/tree/fused_check.py`` is this route's EXHAUSTIVE gate (parity, multi,
goldens, declines, wiring, timing) and stays the authority; it is a dev CLI and is NOT
pytest-collected.  This module is the SUITE's net for the same route: seconds of
assertions that fail loudly when the Triton evaluator (``fused_kernels``), its facade
(``fused_engine``) or the route wiring (``dense_engine.resolve_route`` + ``Forest``)
regresses.  Oracle: ``src/evogp/tests/reference_eval.py`` (read-only, engine-free).

HERMETIC BY CONSTRUCTION: the whole module SKIPS (never fails) when the route cannot
exist -- no CUDA/HIP device visible to torch, ``triton`` not importable, or no writable
Triton JIT cache (``TRITON_CACHE_DIR`` unset on a read-only ``$HOME``).  The decision is
taken at COLLECTION time and imports nothing heavy (``fused_available`` answers from
``importlib.util.find_spec``).  The suite's ``conftest.py`` pins
``EVOGP_EVOXC_BACKEND=numpy`` for hermeticity; the ``_fused_route`` fixture removes that
pin per test, because ``backend="numpy"`` is a documented DECLINE of this route, not a
fallback.

RUN (repo root, torch-carrying venv, writable ``TRITON_CACHE_DIR``)::

    PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1 \\
      TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache \\
      /tmp/rocvenv/bin/python -m pytest src/evogp/tests/test_fused_route.py -q

HOME: this module lives in the SUITE (``src/evogp/tests/``), so its oracle is the
co-located ``reference_eval.py`` and pytest's own ``sys.path`` insertion makes
``import reference_eval`` resolve; the path fallback below only serves a copy run from
elsewhere.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct

import pytest

torch = pytest.importorskip("torch")


def _load_reference():
    """``src/evogp/tests/reference_eval.py``: by import when co-located, else by path."""
    try:
        import reference_eval  # the relocated home: a sibling module

        return reference_eval
    except ImportError:
        pass
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here / "reference_eval.py", here.parent / "tests" / "reference_eval.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("_evogp_fused_reference", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


reference = _load_reference()
if reference is None:  # pragma: no cover - only if the draft is moved without it
    pytest.skip(
        "the plain-torch oracle is not next to this module or in ../tests/",
        allow_module_level=True,
    )

try:  # the route under test, through its public facade only
    from evogp.tree import (
        Forest,
        GenerateDescriptor,
        dense_engine,
        evoxc_engine,
        fused_engine,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"evogp.tree is not importable ({exc!r})", allow_module_level=True)


def _skip_reasons():
    """Why this box cannot run the route (empty list = run the module)."""
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("no CUDA/HIP device is visible to torch")
    if not fused_engine.fused_available():
        reasons.append("the fused route is unavailable (triton not importable, or no device)")
    if not os.environ.get("TRITON_CACHE_DIR", "").strip():
        home = os.path.expanduser("~")
        if not (os.path.isdir(home) and os.access(home, os.W_OK)):
            reasons.append("no writable Triton JIT cache (set TRITON_CACHE_DIR)")
    return reasons


_SKIP = _skip_reasons()
pytestmark = pytest.mark.skipif(
    bool(_SKIP), reason="; ".join(_SKIP) or "device + triton available (never skipped)"
)

DEVICE = torch.device("cuda")
# Node kinds and function ids come from the ORACLE's tables, so a case can never
# disagree with the reference about what an id means.
VAR, CONST, UFUNC, BFUNC, TFUNC = (
    reference.VAR, reference.CONST, reference.UFUNC, reference.BFUNC, reference.TFUNC,
)
OUT_NODE = reference.OUT_NODE
IF, ADD, SUB, MUL, DIV, LOOSE_DIV, POW, LOOSE_POW, MAX, MIN, LT, GT, LE, GE = range(14)
(
    SIN, COS, TAN, SINH, COSH, TANH, LOG, LOOSE_LOG, EXP, INV, LOOSE_INV, NEG, ABS, SQRT,
    LOOSE_SQRT,
) = range(14, 29)

#: ``(using_funcs, rtol)`` per function set.  ``rtol = 0.0`` means the route is BITWISE
#: against the oracle there (measured, ``tree/FUSED.md`` §16.5); the ``1e-6`` sets are the
#: transcendental / loose surfaces, where an emitter may agree in value on a different byte.
VALUE_SETS = {
    "arithmetic": (["+", "-", "*", "/"], 0.0),
    "ternary": (["if", "+", "*"], 0.0),
    "compare_unary": (["<", ">", "<=", ">=", "neg", "abs", "sqrt"], 0.0),
    "transcendental": (["sin", "cos", "tan", "exp", "log", "+", "*"], 1e-6),
    "loose": (["loose_div", "loose_pow", "loose_log", "loose_inv", "loose_sqrt"], 1e-6),
}


# --- tiny tree-spec DSL: nested spec -> prefix-encoded population tensors --------- #
def _node_type_of(function_id):
    """The node kind a function id implies (arity 1/2/3 -> UFUNC/BFUNC/TFUNC)."""
    return {1: UFUNC, 2: BFUNC, 3: TFUNC}[int(reference.arity_of(int(function_id)))]


def _out_node_value(function_id, out_index):
    """``float32`` bit pattern of ``struct OutNodeValue { function, outIndex }``."""
    bits = ((int(out_index) & 0xFFFF) << 16) | (int(function_id) & 0xFF)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _encode(node, output_len):
    """``("var",i) | ("const",v) | ("fn",id,kids) | ("out",id,idx,kids)`` -> prefix."""
    values, types, sizes = [], [], []

    def walk(n):
        tag, index, kids = n[0], len(values), ()
        values.append(0.0)
        types.append(0)
        sizes.append(0)
        if tag == "var":
            values[index], types[index] = float(n[1]), VAR
        elif tag == "const":
            values[index], types[index] = float(n[1]), CONST
        elif tag == "fn":
            values[index], types[index] = float(n[1]), _node_type_of(n[1])
            kids = n[2]
        elif tag == "out":
            assert output_len >= 2, "an OUT_NODE tag needs output_len >= 2"
            values[index] = _out_node_value(n[1], n[2])
            types[index] = _node_type_of(n[1]) | OUT_NODE
            kids = n[3]
        else:
            raise AssertionError(f"unknown tree-spec tag {tag!r}")
        for kid in kids:
            walk(kid)
        sizes[index] = len(values) - index

    walk(node)
    return values, types, sizes


def _forest(specs, *, input_len, output_len=1, pad_to):
    """A device ``Forest`` from nested specs (every row padded to ``pad_to``)."""
    rows = [_encode(spec, output_len) for spec in specs]
    node_value = torch.zeros((len(specs), pad_to), dtype=torch.float32)
    node_type = torch.zeros((len(specs), pad_to), dtype=torch.int16)
    subtree_size = torch.zeros((len(specs), pad_to), dtype=torch.int16)
    for row, (values, types, sizes) in enumerate(rows):
        assert len(values) <= pad_to, f"spec {row} has {len(values)} nodes > pad_to={pad_to}"
        node_value[row, : len(values)] = torch.tensor(values, dtype=torch.float32)
        node_type[row, : len(values)] = torch.tensor(types, dtype=torch.int16)
        subtree_size[row, : len(values)] = torch.tensor(sizes, dtype=torch.int16)
    return Forest(
        input_len, output_len,
        node_value.to(DEVICE), node_type.to(DEVICE), subtree_size.to(DEVICE),
    )


# --- populations, data, oracle ---------------------------------------------------- #
def _random_forest(pop_size, using_funcs, *, input_len=4, output_len=1, max_tree_len=64,
                   max_layer_cnt=None, seed=0):
    """``seed -> descriptor -> population`` (both draw from torch's global RNG)."""
    if max_layer_cnt is None:
        max_layer_cnt = 3 if "if" in using_funcs else 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    descriptor = GenerateDescriptor(
        max_tree_len=max_tree_len, input_len=input_len, output_len=output_len,
        using_funcs=list(using_funcs), max_layer_cnt=max_layer_cnt, layer_leaf_prob=0.2,
        const_prob=0.5, out_prob=0.5, const_range=(0.25, 1.25), sample_cnt=8,
    )
    return Forest.random_generate(pop_size=pop_size, descriptor=descriptor)


def _data(rows, columns, seed=7):
    """Deterministic positive ``(rows, columns)`` device block (LOG/SQRT/DIV safe)."""
    values = torch.rand(rows, columns, generator=torch.Generator().manual_seed(seed))
    return (values + 0.25).to(DEVICE)


def _rows_of(forest):
    return (
        forest.batch_node_value.detach().cpu().float(),
        forest.batch_node_type.detach().cpu(),
        forest.batch_subtree_size.detach().cpu(),
    )


def _oracle(forest, x, *, per_row=False, labels=None, use_mse=True):
    """The independent plain-torch answer for the same tensors."""
    rows, x = _rows_of(forest), x.detach().cpu().float()
    if labels is None:
        emit = reference.forward_from_rows if per_row else reference.batch_forward_from_rows
        return emit(*rows, x, out_len=forest.output_len)
    return reference.sr_fitness_from_rows(
        *rows, x, labels.detach().cpu().float(), use_mse=use_mse, out_len=forest.output_len
    )


# --- assertions: NaN/inf PLACEMENT always exact, then bitwise or within ``rtol`` --- #
def _assert_same(value, expected, context, *, rtol=None):
    """Compare against the oracle; ``rtol=None`` demands bit-identical finite values.

    A NaN *payload* is deliberately not part of the claim (``FUSED.md`` §16.5: the loose
    surface may write a different payload for the same NaN), so non-finite entries are
    pinned by the placement check and only finite ones are compared.
    """
    got = value.detach().to(torch.float32).cpu()
    want = expected.detach().to(torch.float32).cpu()
    assert tuple(got.shape) == tuple(want.shape), (
        f"{context}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    )
    for label, mask, other in (
        ("NaN", torch.isnan(got), torch.isnan(want)),
        ("inf", torch.isinf(got), torch.isinf(want)),
    ):
        assert torch.equal(mask, other), (
            f"{context}: {label} placement differs ({int(mask.sum())} vs {int(other.sum())})"
        )
    both_inf = torch.isinf(got) & torch.isinf(want)
    if bool(both_inf.any()):
        assert torch.equal(torch.sign(got[both_inf]), torch.sign(want[both_inf])), (
            f"{context}: the sign of an infinite entry differs"
        )
    finite = torch.isfinite(got) & torch.isfinite(want)
    if not bool(finite.any()):
        return
    a, b = got[finite].contiguous(), want[finite].contiguous()
    if rtol is None:
        bad, claim = a.view(torch.int32) != b.view(torch.int32), "bit-identical"
    else:
        bad, claim = ~torch.isclose(a, b, rtol=rtol, atol=0.0), f"within rtol={rtol:g}"
    if bool(bad.any()):
        first = int(bad.nonzero()[0])
        rel = float((a[first] - b[first]).abs() / b[first].abs().clamp_min(1e-30))
        raise AssertionError(
            f"{context}: {int(bad.sum())} of {a.numel()} finite values are not {claim}; "
            f"first got={float(a[first])!r} want={float(b[first])!r} (rel={rel:.3e})"
        )


def _deltas(before, after):
    return (
        int(after["fused_declines"] - before["fused_declines"]),
        int(after["fused_kernel_launches"] - before["fused_kernel_launches"]),
    )


def _served(call, context):
    """Run ``call()`` and prove the Triton kernel served it (never a fallback).

    The engine name in a ``Route`` is not a served-call proof: an explicit
    ``EVOGP_EVOXC_BACKEND`` (or a knob) can still decline at call time, so the counters
    are the evidence (``fused_calls`` counts ATTEMPTS, ``fused_kernel_launches`` only
    real launches).
    """
    before = fused_engine.fused_cache_stats()
    got = call()
    declines, launches = _deltas(before, fused_engine.fused_cache_stats())
    assert declines == 0, (
        f"{context}: the fused route DECLINED (+{declines}), so a fallback produced these "
        "numbers -- check EVOGP_EVAL_MODE / EVOGP_FUSED_MIN_POP / "
        "EVOGP_FUSED_MAX_SCRATCH_MB and that EVOGP_EVOXC_BACKEND is not 'numpy'"
    )
    assert launches > 0, (
        f"{context}: no Triton launch was counted -- the call never reached the fused engine"
    )
    return got


def _declined(call, context):
    """Run ``call()`` and prove it was DECLINED (i.e. the fallback served it)."""
    before = fused_engine.fused_cache_stats()
    got = call()
    declines, launches = _deltas(before, fused_engine.fused_cache_stats())
    assert (declines, launches) == (1, 0), (
        f"{context}: expected exactly one decline and no launch, counted "
        f"declines={declines} launches={launches}"
    )
    return got


# --- fixtures --------------------------------------------------------------------- #
@pytest.fixture(scope="module", autouse=True)
def _warm_jit():
    """Compile the four single-output Triton specializations once for the module.

    The JIT cost is per ``(per_row, reduce, use_mse, block, warps)`` combination (~0.4 s
    the first time in a process).  A direct facade call ignores ``EVOGP_EVOXC_BACKEND``,
    which a module-scoped fixture could not neutralise anyway (``monkeypatch`` is
    function-scoped).  The multi-output kernels are compiled by their own test.
    """
    rows = _forest([("fn", ADD, (("var", 0), ("var", 1)))] * 16, input_len=4, pad_to=8)
    call = (rows.batch_node_value, rows.batch_node_type, rows.batch_subtree_size)
    broadcast, labels, per_row = _data(8, 4, seed=1), _data(8, 1, seed=3), _data(16, 4, 2)
    fused_engine.evaluate_fused(*call, 4, 1, broadcast, per_row=False)
    fused_engine.evaluate_fused(*call, 4, 1, per_row, per_row=True)
    for use_mse in (True, False):
        fused_engine.sr_fitness_fused(*call, 4, 1, broadcast, labels, use_MSE=use_mse)
    yield


@pytest.fixture(autouse=True)
def _fused_route(monkeypatch):
    """Point the PUBLIC API at the fused engine for the duration of the test.

    ``EVOGP_EVAL_MODE=fused`` selects the route; the evoxc backend pin the suite's
    ``conftest.py`` sets must go, because ``backend="numpy"`` is a documented DECLINE of
    this route (Triton has no numpy twin), not a fallback.
    """
    monkeypatch.setenv("EVOGP_EVAL_MODE", "fused")
    for name in ("EVOGP_EVOXC_BACKEND", "EVOGP_FUSED_MIN_POP", "EVOGP_FUSED_MAX_SCRATCH_MB",
                 "EVOGP_FUSED_BLOCK_ROWS", "EVOGP_FUSED_NUM_WARPS",
                 "EVOGP_FUSED_DISPATCH"):
        monkeypatch.delenv(name, raising=False)
    yield


# --- 1. value-path parity through the public API ---------------------------------- #
@pytest.mark.parametrize("per_row", [False, True], ids=["batch_forward", "forward"])
@pytest.mark.parametrize("name", sorted(VALUE_SETS))
def test_value_parity(name, per_row):
    """``batch_forward`` (per_row=False) and ``forward`` (per_row=True) vs the oracle."""
    using_funcs, rtol = VALUE_SETS[name]
    forest = _random_forest(24, using_funcs)
    x = _data(forest.pop_size if per_row else 16, forest.input_len, seed=13 if per_row else 11)
    context = f"{'forward' if per_row else 'batch_forward'}[{name}]"
    got = _served(
        lambda: forest.forward(x) if per_row else forest.batch_forward(x), context
    )
    want = _oracle(forest, x, per_row=per_row)
    _assert_same(got, want, context, rtol=None if rtol == 0.0 else rtol)


def test_single_node_and_constant_populations():
    """A VAR root, a CONST root and a fully constant population."""
    x = _data(8, 4)
    cases = {
        "var_root": [("var", 1)],
        "const_root": [("const", 2.5)],
        "constant_population": [
            ("fn", ADD, (("const", 1.5), ("fn", MUL, (("const", 2.0), ("const", 3.0)))))
        ],
    }
    for name, specs in cases.items():
        forest = _forest(specs * 16, input_len=4, pad_to=8)
        context = f"batch_forward[{name}]"
        got = _served(lambda: forest.batch_forward(x), context)
        _assert_same(got, _oracle(forest, x), context)
        per_row = _data(forest.pop_size, 4, seed=21)
        _assert_same(
            forest.forward(per_row), _oracle(forest, per_row, per_row=True), f"forward[{name}]"
        )


def test_deep_chain():
    """An 81-node deep chain (well past 40): the stack discipline under load."""
    chain = ("var", 0)
    for level in range(1, 41):
        chain = ("fn", ADD, (chain, ("var", level % 4)))
    forest = _forest([chain] * 16, input_len=4, pad_to=128)
    length = int(forest.batch_subtree_size[0, 0])
    assert length >= 40, f"the deep-chain case regressed to {length} nodes"
    x = _data(8, 4)
    got = _served(lambda: forest.batch_forward(x), "batch_forward[deep_chain]")
    _assert_same(got, _oracle(forest, x), f"batch_forward[deep_chain {length} nodes]")
    per_row = _data(forest.pop_size, 4, seed=22)
    _assert_same(
        forest.forward(per_row), _oracle(forest, per_row, per_row=True), "forward[deep_chain]"
    )


def test_protected_div_and_nan_inf_placement():
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
    forest = _forest([spec for spec in edge for _ in range(2)], input_len=4, pad_to=32)
    x = _data(8, 4)
    got = _served(lambda: forest.batch_forward(x), "batch_forward[edge]")
    want = _oracle(forest, x)
    assert int(torch.isnan(want).sum()) > 0, (
        "the edge population produces no NaN any more, so it stopped exercising NaN placement"
    )
    _assert_same(got, want, "batch_forward[edge: div/0, log(0), sqrt(-1), overflow]")
    per_row = _data(forest.pop_size, 4, seed=23)
    _assert_same(forest.forward(per_row), _oracle(forest, per_row, per_row=True), "forward[edge]")


# --- 2. loss path and its result contract ----------------------------------------- #
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
@pytest.mark.parametrize("name", ["arithmetic", "loose"])
def test_sr_fitness_matches_oracle(name, use_mse):
    """``Forest.SR_fitness`` vs the oracle's reduction, for both loss metrics."""
    forest = _random_forest(24, VALUE_SETS[name][0])
    inputs, labels = _data(16, forest.input_len), _data(16, 1, seed=31)
    context = f"SR_fitness[{name}, mse={use_mse}]"
    loss = _served(lambda: forest.SR_fitness(inputs, labels, use_MSE=use_mse), context)
    _assert_same(loss, _oracle(forest, inputs, labels=labels, use_mse=use_mse), context, rtol=1e-6)


def test_sr_fitness_result_contract():
    """``(pop,)`` float32, contiguous, no grad, on the device, and a FRESH buffer."""
    forest = _random_forest(24, ["+", "*"])
    inputs, labels = _data(16, forest.input_len), _data(16, 1, seed=41)
    loss = _served(lambda: forest.SR_fitness(inputs, labels), "SR_fitness[contract]")
    assert tuple(loss.shape) == (forest.pop_size,), f"shape {tuple(loss.shape)}"
    assert loss.dtype == torch.float32, f"dtype {loss.dtype}"
    assert loss.is_contiguous(), "the pipeline mutates the loss in place: it must be contiguous"
    assert loss.requires_grad is False, "the loss must not carry an autograd graph"
    assert loss.device.type == "cuda", f"the loss lives on {loss.device}"
    snapshot = loss.clone()
    loss.fill_(-123.0)  # the pipeline writes into the result; the next call must be clean
    _assert_same(forest.SR_fitness(inputs, labels), snapshot, "SR_fitness[fresh buffer]")


# --- 3. multi-output (OUT_NODE-tagged) populations --------------------------------- #
def _tagged(function_id, out_index, children):
    return ("out", function_id, out_index, children)


def _multi_population():
    """Four tagged shapes x 4: two outputs, a tagged ROOT, one tagged, same output twice."""
    two_outputs = ("fn", ADD, (
        _tagged(NEG, 0, (("var", 0),)), _tagged(ADD, 1, (("var", 1), ("var", 2))),
    ))
    root_if = _tagged(IF, 1, (
        ("var", 0),
        ("fn", ADD, (("var", 1), ("const", 0.5))),
        ("fn", MUL, (("var", 2), ("var", 3))),
    ))
    one_tagged = ("fn", ADD, (
        _tagged(MUL, 1, (("var", 0), ("const", 2.5))), ("fn", SUB, (("var", 1), ("var", 2))),
    ))
    same_output = _tagged(ADD, 0, (("var", 0), _tagged(NEG, 0, (("var", 1),))))
    return [two_outputs, root_if, one_tagged, same_output] * 4


@pytest.mark.parametrize("out_len", [2, 3])
def test_multi_output_value_parity(out_len):
    """``batch_forward`` and ``forward`` at ``output_len`` 2 and 3: BITWISE."""
    forest = _forest(_multi_population(), input_len=4, output_len=out_len, pad_to=32)
    x, context = _data(16, 4), f"multi_batch_forward[out_len={out_len}]"
    got = _served(lambda: forest.batch_forward(x), context)
    _assert_same(got, _oracle(forest, x), context)
    per_row = _data(forest.pop_size, 4, seed=33)
    _assert_same(
        forest.forward(per_row),
        _oracle(forest, per_row, per_row=True),
        f"forward[multi, out_len={out_len}]",
    )


@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
@pytest.mark.parametrize("out_len", [2, 3])
def test_multi_output_loss_parity(out_len, use_mse):
    """``SR_fitness`` on tagged trees: the documented ``<= 1e-6`` relative, NOT bitwise.

    The in-kernel reduction sums the per-output errors in fp32 in its own order while the
    oracle sums with torch (measured ~1e-7 relative -- the ``LOSS_RTOL`` story of
    ``tree/FUSED.md`` §16.5).
    """
    forest = _forest(_multi_population(), input_len=4, output_len=out_len, pad_to=32)
    inputs, labels = _data(16, 4), _data(16, out_len, seed=35)
    context = f"SR_fitness[multi, out_len={out_len}, mse={use_mse}]"
    loss = _served(lambda: forest.SR_fitness(inputs, labels, use_MSE=use_mse), context)
    assert tuple(loss.shape) == (forest.pop_size,), f"shape {tuple(loss.shape)}"
    _assert_same(loss, _oracle(forest, inputs, labels=labels, use_mse=use_mse), context, rtol=1e-6)


# --- 3b. the OUT_NODE refusal (a single-output request over tagged rows) ----------- #
def _miswired_forest():
    """REAL ``output_len = 2`` rows handed to the engine as a single-output population.

    This is what a wrong ``GenerateDescriptor.output_len`` / ``Forest(input_len, 1, ...)``
    produces: the tags on the rows are the out-nodes' own.  Before the shared rule
    (``evogp.tree.multi_output``) the fused kernel read such a type as a ternary ``if``
    and returned a plausible wrong number; the facade now refuses it host-side.
    """
    genuine = _forest(_multi_population(), input_len=4, output_len=2, pad_to=32)
    return Forest(
        4,
        1,
        genuine.batch_node_value,
        genuine.batch_node_type,
        genuine.batch_subtree_size,
    )


def test_refuse_out_node_tags_in_single_output_mode():
    """The direct facade refuses, and it is NOT a decline: nothing is launched."""
    forest = _miswired_forest()
    assert bool((forest.batch_node_type & OUT_NODE).any()), "the fixture lost its tags"
    call = (forest.batch_node_value, forest.batch_node_type, forest.batch_subtree_size)
    broadcast, labels = _data(16, 4), _data(16, 1, seed=41)
    per_row = _data(forest.pop_size, 4, seed=43)
    before = fused_engine.fused_cache_stats()
    for context, thunk in (
        ("evaluate_fused", lambda: fused_engine.evaluate_fused(*call, 4, 1, broadcast, per_row=False)),
        (
            "evaluate_fused[per_row]",
            lambda: fused_engine.evaluate_fused(*call, 4, 1, per_row, per_row=True),
        ),
        (
            "sr_fitness_fused",
            lambda: fused_engine.sr_fitness_fused(*call, 4, 1, broadcast, labels),
        ),
    ):
        with pytest.raises(ValueError, match="OUT_NODE") as caught:
            thunk()
        assert str(caught.value).startswith("row 0: node "), (
            f"{context}: the refusal must name row and node, got {str(caught.value)!r}"
        )
    declines, launches = _deltas(before, fused_engine.fused_cache_stats())
    assert (declines, launches) == (0, 0), (
        "the OUT_NODE refusal must happen before any launch and must not be counted as a "
        f"DECLINE (a decline falls back and serves): declines={declines} launches={launches}"
    )


def test_refuse_out_node_tags_through_the_public_api(monkeypatch):
    """``EVOGP_EVAL_MODE=fused`` and ``auto`` raise ``ValueError`` -- never a number."""
    for mode in ("fused", "auto"):
        monkeypatch.setenv("EVOGP_EVAL_MODE", mode)
        forest, x = _miswired_forest(), _data(16, 4)
        with pytest.raises(ValueError, match="OUT_NODE"):
            forest.batch_forward(x)
        with pytest.raises(ValueError, match="OUT_NODE"):
            forest.SR_fitness(x, _data(16, 1, seed=47))
        with pytest.raises(ValueError, match="OUT_NODE"):
            forest.forward(_data(forest.pop_size, 4, seed=49))


def test_the_same_rows_are_served_once_output_len_says_two():
    """The refusal is scoped to ``output_len == 1``: the tagged rows still evaluate."""
    genuine = _forest(_multi_population(), input_len=4, output_len=2, pad_to=32)
    x = _data(16, 4)
    got = _served(lambda: genuine.batch_forward(x), "batch_forward[multi, out_len=2]")
    _assert_same(got, _oracle(genuine, x), "batch_forward[multi, out_len=2]")


# --- 4. route wiring --------------------------------------------------------------- #
@pytest.mark.parametrize("pop_size", [1, 16, 64, 256])
def test_route_policy_selects_fused(pop_size):
    """``EVOGP_EVAL_MODE=fused`` routes to fused from ``EVOGP_FUSED_MIN_POP`` up."""
    minimum = fused_engine.resolve_min_pop()
    route = dense_engine.resolve_route(pop_size)
    expected = "specialized" if pop_size < minimum else "fused"
    assert route.engine == expected, (
        f"pop_size={pop_size} against min_pop={minimum}: expected {expected!r}, got "
        f"{route.engine!r} ({route.reason})"
    )
    if expected == "fused":
        assert route.use_dense is False
        assert (route.backend, route.fallback_backend) == ("iree", "numpy"), (
            "the fused route must run on a device backend with a specialized+numpy "
            f"fallback, got {route.backend!r} / {route.fallback_backend!r}"
        )


@pytest.mark.parametrize("pop_size", [16, 64, 256])
def test_auto_selects_fused_above_the_floor(pop_size, monkeypatch):
    """``auto`` -- the DEFAULT -- owns the fused route from ``EVOGP_FUSED_MIN_POP`` up.

    The pre-policy contract this file used to pin ("``auto`` never selects fused") is
    gone: on a device+triton box (the only box this module runs on) the default
    resolves ``engine == "fused"`` from the floor up, on the ``iree`` backend
    SPELLING with the historical specialized+``numpy`` fallback.  The REAL probes are
    the oracle here -- nothing is pinned, so a rolled-back policy cannot pass.
    The rest of the matrix (the floor moving, the no-triton / no-device worlds, the
    fused envelope, the explicit-backend arm that never reaches fused) is pinned once
    in ``test_route_policy.py`` section 6; this is the fused file's own guard that the
    default did not silently go back.
    """
    monkeypatch.delenv("EVOGP_EVAL_MODE", raising=False)
    minimum = fused_engine.resolve_min_pop()
    assert minimum == fused_engine.DEFAULT_MIN_POP == 16, "the documented crossover"
    assert pop_size >= minimum
    assert torch.cuda.is_available() and fused_engine.fused_available(), (
        "the module-level skip guarantees device + triton on this box"
    )
    route = dense_engine.resolve_route(pop_size)
    assert route.engine == "fused", (pop_size, route.reason)
    assert route.use_fused is True and route.use_dense is False
    assert (route.backend, route.fallback_backend) == ("iree", "numpy"), (
        f"pop_size={pop_size}: got {route.backend!r} / {route.fallback_backend!r}"
    )


def test_auto_keeps_the_pre_fused_policy_below_the_floor_or_without_triton(monkeypatch):
    """The two guards of that default: the pop floor, and the availability probe.

    Below ``EVOGP_FUSED_MIN_POP`` nothing moved -- specialized+``numpy``, the
    ``Tree`` / small-population answer, because a cold Triton JIT cannot amortize
    there -- and with the fused probe off ``auto`` is the historical
    dense+iree-then-specialized rule verbatim.
    """
    monkeypatch.delenv("EVOGP_EVAL_MODE", raising=False)
    floor = fused_engine.resolve_min_pop()
    # the probe is ON here (the module-level skip guarantees it), so the answers
    # below are the FLOOR's doing and not a silently unavailable route
    assert dense_engine.fused_route_available() is True
    for pop_size in (1, 8, floor - 1):
        route = dense_engine.resolve_route(pop_size)
        assert route.engine == "specialized", (pop_size, route.reason)
        assert route.use_fused is False and route.use_dense is False
        assert (route.backend, route.fallback_backend) == ("numpy", "numpy"), pop_size
    assert dense_engine.resolve_route(floor).engine == "fused", "one tree more, fused"
    # the pre-change world: the probe off (`triton` missing, or no device)
    monkeypatch.setattr(dense_engine, "_FUSED_AVAILABLE", False)
    for pop_size in (64, 256):
        route = dense_engine.resolve_route(pop_size)
        assert route.engine == "dense", (pop_size, route.reason)
        assert route.use_dense is True and route.use_fused is False
        assert (route.backend, route.fallback_backend) == ("iree", "numpy"), pop_size
    assert dense_engine.resolve_route(floor - 1).engine == "specialized"


def test_served_call_increments_launches_only():
    """A served call bumps ``fused_kernel_launches`` and ``fused_calls`` only."""
    forest = _random_forest(32, ["+", "*"])
    x = _data(8, forest.input_len)
    before = fused_engine.fused_cache_stats()
    forest.batch_forward(x)
    after = fused_engine.fused_cache_stats()
    calls = int(after["fused_calls"] - before["fused_calls"])
    assert calls == 1, f"one public-API call must be exactly one fused call, counted {calls}"
    assert _deltas(before, after) == (0, 1), (
        f"one served call must launch once and never decline, counted decl={_deltas(before, after)}"
    )


# --- 5. declines: the documented "fall back to specialized+numpy" contract -------- #
def test_decline_wide_output_len(monkeypatch):
    """``output_len`` above ``MAX_OUT_LEN`` declines and the fallback serves it."""
    out_len = int(fused_engine.MAX_OUT_LEN) + 1
    forest = _forest(
        [_tagged(ADD, 0, (("var", 0), ("var", 1)))] * 16,
        input_len=4, output_len=out_len, pad_to=32,
    )
    x = _data(8, 4)
    seen = {}
    original = evoxc_engine.evaluate

    def spy(population, inputs, **kwargs):
        seen["backend"] = kwargs.get("backend")
        return original(population, inputs, **kwargs)

    monkeypatch.setattr(evoxc_engine, "evaluate", spy)
    got = _declined(lambda: forest.batch_forward(x), f"batch_forward[out_len={out_len}]")
    _assert_same(got, _oracle(forest, x), f"decline[out_len={out_len}] fallback values")
    assert seen.get("backend") == "numpy", (
        "the fallback must be the specialized engine on numpy (a per-tree iree compile "
        f"cannot amortize after a decline); recorded backend={seen.get('backend')!r}"
    )


def test_decline_tiny_scratch_budget(monkeypatch):
    """A scratch budget below the launch's real bytes declines, then falls back.

    The envelope is the bytes one launch ALLOCATES (``tree/FUSED.md`` §12):
    ``pop * n_rb * tree_len * block * 4`` with ``block`` = the largest power of two
    ``<= min(rows, cap)``, never ``rows`` itself.  This call is 32 trees x 64 cells at 160
    data rows, i.e. 2 blocks of 128 lanes -> 32 * 2 * 64 * 128 * 4 B = 4 MiB of scratch,
    comfortably above the 1 MiB budget below (16 rows would be a 16-lane block, i.e.
    128 KiB, and would be SERVED).
    """
    forest = _random_forest(32, ["+", "*"])
    x = _data(160, forest.input_len)
    baseline = forest.batch_forward(x)
    monkeypatch.setenv("EVOGP_FUSED_MAX_SCRATCH_MB", "1")
    got = _declined(lambda: forest.batch_forward(x), "batch_forward[budget=1MB]")
    _assert_same(got, baseline, "decline[scratch budget] fallback values")


# --- 6. guards for the defects the hardening round fixed -------------------------- #
def test_noncontiguous_inputs_are_read_correctly():
    """A column slice of a padded ``x`` (and of the labels) equals its contiguous copy."""
    forest = _random_forest(24, ["+", "*"])
    sliced = _data(16, forest.input_len + 8)[:, : forest.input_len]
    assert not sliced.is_contiguous(), "the case stopped being a non-contiguous view"
    _assert_same(
        forest.batch_forward(sliced),
        forest.batch_forward(sliced.contiguous()),
        "batch_forward[non-contiguous x]",
    )
    row_slice = _data(forest.pop_size, forest.input_len + 8, seed=17)[:, : forest.input_len]
    assert not row_slice.is_contiguous()
    _assert_same(
        forest.forward(row_slice),
        forest.forward(row_slice.contiguous()),
        "forward[non-contiguous x]",
    )
    labels = _data(16, 3, seed=19)[:, :1]
    assert not labels.is_contiguous()
    _assert_same(
        forest.SR_fitness(sliced, labels),
        forest.SR_fitness(sliced.contiguous(), labels.contiguous()),
        "SR_fitness[non-contiguous inputs/labels]",
    )


def test_noncontiguous_population_equals_contiguous():
    """A stride-2 population table equals its contiguous copy, bitwise."""
    forest = _random_forest(24, ["+", "*"])
    x = _data(8, forest.input_len)

    def strided(table):
        wide = torch.zeros(
            (table.shape[0], 2 * table.shape[1]), dtype=table.dtype, device=table.device
        )
        wide[:, ::2] = table
        return wide[:, ::2]

    view = Forest(
        forest.input_len, forest.output_len,
        strided(forest.batch_node_value), strided(forest.batch_node_type),
        strided(forest.batch_subtree_size),
    )
    assert not view.batch_subtree_size.is_contiguous(), (
        "the case stopped being a non-contiguous population"
    )
    got = _served(lambda: view.batch_forward(x), "batch_forward[stride-2 population]")
    _assert_same(got, forest.batch_forward(x), "batch_forward[stride-2 population]")


def test_subtree_size_clamp():
    """An oversized / negative ``subtree_size[0]`` must not crash or spill."""
    spec = ("fn", ADD, (("fn", MUL, (("var", 0), ("const", 1.5))), ("var", 1)))
    honest = _forest([spec] * 16, input_len=4, pad_to=16)
    x = _data(8, 4)
    honest_values = honest.batch_forward(x)

    oversized = honest.batch_subtree_size.clone()
    oversized[0, 0] = 30000  # clamps to tree_len: the honest row's own answer
    _assert_same(
        Forest(4, 1, honest.batch_node_value, honest.batch_node_type, oversized).batch_forward(x),
        honest_values,
        "clamp[subtree_size[0]=30000]",
    )

    negative = honest.batch_subtree_size.clone()
    negative[0, 0] = -5  # floors to 0: the walk does nothing, the value is exactly 0
    floored = Forest(4, 1, honest.batch_node_value, honest.batch_node_type, negative)
    floored_values = floored.batch_forward(x)
    assert float(floored_values[0].abs().max()) == 0.0, (
        "a negative subtree_size[0] must floor to an empty walk (0.0), got "
        f"max|value|={float(floored_values[0].abs().max()):.3e}"
    )
    _assert_same(
        floored_values[1:].unsqueeze(0),
        honest_values[1:].unsqueeze(0),
        "clamp[subtree_size[0]=-5] leaves the other rows alone",
    )


def test_clear_cache_resets_the_fused_tables():
    """``evoxc_engine.clear_cache()`` is the one in-memory reset, fused tables included."""
    forest = _random_forest(24, ["+", "*"])
    x = _data(8, forest.input_len)
    forest.batch_forward(x)  # make sure every counter has moved
    evoxc_engine.clear_cache()
    for key in ("fused_calls", "fused_declines", "fused_kernel_launches", "fused_xt_entries"):
        assert fused_engine.fused_cache_stats()[key] == 0, f"{key} survived clear_cache()"
    got = _served(lambda: forest.batch_forward(x), "batch_forward[after clear_cache]")
    _assert_same(got, _oracle(forest, x), "batch_forward[after clear_cache]")


# --- 7. the launch path (``EVOGP_FUSED_DISPATCH``: pre-bound vs ``JITFunction.run``) --- #
# The route's per-call cost at the small cells is dominated by Triton's PYTHON
# dispatcher, not by the kernel (``FUSED.md`` §18), so the shipped path binds the
# ``CompiledKernel`` once per prepared call and serves every later call directly.
# These cases pin the contract that makes that safe: the knob's spellings, the
# counters, bitwise equality with the historical dispatch, the re-arm point and the
# two declines.
def _dispatch_stats():
    """``fused_kernels.dispatch_stats()``, i.e. the launch path's own counters."""
    from evogp.tree import fused_kernels

    return fused_kernels.dispatch_stats()


def test_resolve_dispatch_spellings(monkeypatch):
    """``bound`` is the default, ``jit`` the A/B arm, and a typo is a caller error."""
    from evogp.tree import fused_kernels

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


def test_jit_mode_serves_every_call_through_the_dispatcher(monkeypatch):
    """``EVOGP_FUSED_DISPATCH=jit`` reproduces the pre-change dispatch exactly."""
    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "jit")
    fused_engine.clear_fused_cache()
    forest = _random_forest(16, ["+", "*"])
    x = _data(8, forest.input_len)
    got = [forest.batch_forward(x) for _ in range(3)]
    for index, value in enumerate(got):
        _assert_same(value, _oracle(forest, x), f"batch_forward[jit mode, call {index}]")
    stats, launches = _dispatch_stats(), fused_engine.fused_cache_stats()
    assert stats["bound_launches"] == 0, (
        f"jit mode must never serve a pre-bound launch, counted {stats}"
    )
    assert stats["jit_launches"] == int(launches["fused_kernel_launches"]) == 3, (
        "every call of a jit-mode plan pays the dispatcher (3 calls, 3 dispatches); "
        f"dispatch={stats} launches={launches['fused_kernel_launches']}"
    )


def test_bound_mode_serves_warm_calls_after_one_dispatch():
    """The default binds the kernel on the first call and skips the dispatcher after."""
    fused_engine.clear_fused_cache()
    forest = _random_forest(16, ["+", "*"])
    x = _data(8, forest.input_len)
    outputs = [forest.batch_forward(x) for _ in range(4)]
    stats = _dispatch_stats()
    assert stats["declines"] == 0, (
        f"the pre-bound dispatch declined on this box: {stats} -- the route silently "
        "served the historical path, so the timings would be the old ones"
    )
    assert (stats["jit_launches"], stats["bound_launches"]) == (1, 3), (
        "one prepared call = ONE dispatcher pass, then the pre-bound kernel; counted "
        f"{stats}"
    )
    for index, value in enumerate(outputs):
        _assert_same(value, _oracle(forest, x), f"batch_forward[bound mode, call {index}]")
    assert stats["bound_launches"] + stats["jit_launches"] == int(
        fused_engine.fused_cache_stats()["fused_kernel_launches"]
    ), "every served call is exactly one launch, bound or dispatched"


def test_bound_and_jit_agree_bitwise(monkeypatch):
    """The two launch paths are the SAME device code: value, loss and per-row agree."""
    fused_engine.clear_fused_cache()
    forest = _random_forest(24, ["+", "*", "sin"])
    x, labels = _data(12, forest.input_len), _data(12, 1, seed=5)
    per_row = _data(forest.pop_size, forest.input_len, seed=11)

    def run_all():
        # TWICE: one prepared call per surface, then one WARM call per surface -- the
        # warm half is where the two launch paths differ.
        run_all_once()
        return run_all_once()

    def run_all_once():
        return (
            forest.batch_forward(x),
            forest.SR_fitness(x, labels),
            forest.forward(per_row),
        )

    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "jit")
    fused_engine.clear_fused_cache()
    jit_value, jit_loss, jit_per_row = run_all()
    jit_stats = _dispatch_stats()

    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "bound")
    fused_engine.clear_fused_cache()
    bound_value, bound_loss, bound_per_row = run_all()
    bound_stats = _dispatch_stats()

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
    forest = _random_forest(16, ["+", "*"])
    x = _data(8, forest.input_len)
    first = forest.batch_forward(x)
    forest.batch_forward(x)
    before = _dispatch_stats()
    assert (before["jit_launches"], before["bound_launches"]) == (1, 1), before

    fused_engine.clear_fused_cache()
    after_clear = _dispatch_stats()
    assert after_clear == {"bound_launches": 0, "jit_launches": 0, "declines": 0}, (
        f"the launch-path counters must reset with the route's tables, got {after_clear}"
    )
    again = forest.batch_forward(x)
    _assert_same(again, first, "batch_forward[after clear_fused_cache]")
    assert _dispatch_stats()["jit_launches"] == 1, (
        "a cleared plan must re-arm (one dispatcher pass) instead of reusing the "
        "previous binding"
    )


def test_active_launch_hooks_decline(monkeypatch):
    """A registered Triton launch hook keeps the historical dispatch (profilers)."""
    triton_knobs = pytest.importorskip("triton.knobs")
    hook = triton_knobs.HookChain()
    seen = []
    hook.add(lambda *args, **kwargs: seen.append(1))
    monkeypatch.setattr(triton_knobs.runtime, "launch_enter_hook", hook)
    fused_engine.clear_fused_cache()
    forest = _random_forest(16, ["+", "*"])
    x = _data(8, forest.input_len)
    got = [forest.batch_forward(x) for _ in range(3)]
    stats = _dispatch_stats()
    assert stats["declines"] >= 1, (
        "an ACTIVE launch hook must decline the pre-bound path (it would skip the "
        f"hook); counted {stats}"
    )
    assert stats["bound_launches"] == 0, stats
    assert seen, "the hook itself must have run (the C launcher still calls it)"
    for index, value in enumerate(got):
        _assert_same(value, _oracle(forest, x), f"batch_forward[hooked, call {index}]")


def test_unusable_launcher_declines(monkeypatch):
    """A backend launcher of another shape keeps the historical dispatch, silently."""
    from evogp.tree import fused_kernels

    monkeypatch.setattr(fused_kernels, "_launcher_takes_launch_args", lambda launcher: False)
    fused_engine.clear_fused_cache()
    forest = _random_forest(16, ["+", "*"])
    x = _data(8, forest.input_len)
    got = [forest.batch_forward(x) for _ in range(3)]
    stats = _dispatch_stats()
    assert stats["bound_launches"] == 0 and stats["declines"] >= 1, stats
    for index, value in enumerate(got):
        _assert_same(value, _oracle(forest, x), f"batch_forward[declined, call {index}]")
