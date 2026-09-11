"""Engine-vs-reference equivalence tests for the ETL-based ("true jit") EvoGP evaluator.

Purpose
-------
The five custom CUDA ops (``torch.ops.evogp_cuda.tree_generate``/``tree_evaluate``/
``tree_SR_fitness``/``tree_mutate``/``tree_crossover``) are replaced by

* an ETL-based evaluation engine (host-side decode of the population tensors ->
  ``etl.build_cached`` -> shared ``etl.CompileCache``), and
* plain-torch tensor ops for generation/mutation/crossover.

This module checks the new engine against an INDEPENDENT oracle (``reference_eval.py``,
a plain-torch re-implementation of the original ``cuda/forward.cu`` semantics) through
the public ``evogp.tree`` API only:

    Forest.random_generate / Forest(...) / Forest.forward / Forest.batch_forward /
    Forest.SR_fitness / Tree.forward / Tree.SR_fitness

Nothing here imports ``evogp.evogp_cuda``. When ``evogp.tree`` cannot be imported at
all (extension still missing / rewrite not merged) the engine-dependent tests SKIP
with the import error as the reason; the ``test_reference_*`` tests (which calibrate
the oracle against hand-computed values derived from ``cuda/forward.cu``) still run.

How to run (from the repository root)
-------------------------------------
    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q

Environment knobs
-----------------
* ``EVOGP_EQ_TREES``  number of randomly generated trees per population *per seed*
                      (default 1000)
* ``EVOGP_EQ_N``      number of data points used by the large comparisons (default 128)
* ``EVOGP_EQ_SEEDS``  comma-separated list of seeds (default ``0,1,2,3,4,5,6,7``);
                      the legacy single ``EVOGP_EQ_SEED`` is accepted and means a
                      one-element list
* ``EVOGP_EQ_DEVICE`` device for the engine tensors (default: cuda if available else cpu)
* ``EVOGP_GOLDEN_DIR`` extra directory scanned for golden fixtures (see below)

Determinism
-----------
Every randomized comparison seeds the global CPU **and** CUDA torch RNGs
(``torch.manual_seed`` + ``torch.cuda.manual_seed_all``) *before* the
``GenerateDescriptor`` is built and before the population is generated, and runs
over the fixed seed list above. The descriptor's ``const_samples`` are drawn from
the global CUDA RNG at construction time, so seeding after the descriptor was
built (as an earlier revision did) made these tests dependent on whatever RNG
state the previously executed test happened to leave behind -- reproducible only
by accident. Sample data is generated with explicit ``torch.Generator``s.

Numeric contract of the comparisons
-----------------------------------
The oracle (``reference_eval.py``) is a stepwise *correctly rounded float32*
evaluator: every node value is the correctly rounded float32 result of its op on
float32 operands (float64 intermediate for the transcendentals whose float32
kernels are not correctly rounded). Divergences between the engine and the
oracle are therefore only the *rounding* differences of two implementations of
the same correctly rounded function, and any amplification of them by a
cancelling / dividing parent node is a property of the tree, not of the engine.

Golden fixtures (old-CUDA cross-check)
--------------------------------------
``*.npz`` files are scanned, in order, in ``$EVOGP_GOLDEN_DIR``,
``<repo-root>/test/goldens/`` and ``src/evogp/tests/goldens/``. Two independent
schemas are recognized; files matching neither are SKIPPED with an explicit
reason (see below). Everything here degrades to a clean ``pytest.skip`` when the
fixtures (or their individual arrays) are absent.

*a) engine schema* -- ``src/evogp/tests/goldens/*.npz``, captured from this engine.
One ``.npz`` per case with:

* ``node_value``  (pop, max_tree_len) float32
* ``node_type``   (pop, max_tree_len) int16
* ``subtree_size`` (pop, max_tree_len) int16
* ``x``           (N, input_len) float32
* ``expected``    (pop, N, out_len) float32   -- optional, checked against batch_forward
* ``expected_fitness`` (pop,) float32         -- optional, checked against SR_fitness
* ``labels``      (N, out_len) float32        -- optional, needed by expected_fitness
* ``use_mse``     scalar bool                 -- optional, default True (MAE when false)

*b) legacy-capture schema* -- ``test/goldens/*.npz``, the frozen outputs of the
deleted custom CUDA kernels (``test/golden_capture.py``; array names, semantics
and tolerances are documented in ``test/goldens/README.md``). The population is
stored per stage as ``forest__node_value``/``forest__node_type``/
``forest__subtree_size`` and the expectations as ``tree_forward`` /
``forest_forward`` / ``forest_batch_forward`` / ``forest_sr_fitness__<mode>__<mse|nonmse>``
/ ``tree_sr_fitness__*`` / ``problem_evaluate__*``. Rebuilding the ``Forest`` from
those arrays involves no RNG, so this is the strongest available cross-check of
the new engine against the kernels it replaced (floats rtol=atol=1e-4, structural
arrays exact, NaN/±inf placement exact).

Anything else in a golden directory is reported through a ``warnings.warn`` (with
the keys that were found and the keys that were expected) and never fails the suite.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest
import torch

# --------------------------------------------------------------------------- #
# The oracle (must stay independent from the engine)
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import reference_eval as reference
except ImportError:  # pragma: no cover - only if pytest uses importlib import mode
    import importlib.util

    _spec = importlib.util.spec_from_file_location("evogp_reference_eval", _HERE / "reference_eval.py")
    reference = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(reference)

_REPO_ROOT = _HERE.parents[2]

# --------------------------------------------------------------------------- #
# Engine under test (public API only)
# --------------------------------------------------------------------------- #
try:
    from evogp.tree import Forest, Tree, GenerateDescriptor, MAX_STACK, randint, NType

    _ENGINE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the environment
    Forest = Tree = GenerateDescriptor = MAX_STACK = randint = NType = None
    _ENGINE_IMPORT_ERROR = exc

requires_engine = pytest.mark.skipif(
    _ENGINE_IMPORT_ERROR is not None,
    reason=f"evogp.tree is not importable ({_ENGINE_IMPORT_ERROR!r}) -- "
    "the ETL engine / plain-torch operators are expected here; "
    "run the suite again after they are merged.",
)

# --------------------------------------------------------------------------- #
# Problem layer (needed by the problem-level golden comparisons only)
# --------------------------------------------------------------------------- #
try:
    from evogp.problem import SymbolicRegression

    _PROBLEM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the environment
    SymbolicRegression = None
    _PROBLEM_IMPORT_ERROR = exc

requires_problem = pytest.mark.skipif(
    _PROBLEM_IMPORT_ERROR is not None,
    reason=f"evogp.problem is not importable ({_PROBLEM_IMPORT_ERROR!r}); "
    "the problem-level golden comparisons need it",
)
def _parse_seeds() -> tuple:
    """Fixed seed list: ``EVOGP_EQ_SEEDS="0,2,5"`` (legacy: ``EVOGP_EQ_SEED=0``)."""
    raw = os.environ.get("EVOGP_EQ_SEEDS")
    if raw is None:
        raw = os.environ.get("EVOGP_EQ_SEED", "0,1,2,3,4,5,6,7")
    seeds = tuple(int(part) for part in raw.replace(" ", "").split(",") if part)
    assert seeds, f"EVOGP_EQ_SEEDS={raw!r} does not contain a seed"
    return seeds


TREES = int(os.environ.get("EVOGP_EQ_TREES", "1000"))
N_POINTS = int(os.environ.get("EVOGP_EQ_N", "128"))
SEEDS = _parse_seeds()
SEED = SEEDS[0]  # master seed for the deterministic sample data
DEVICE = os.environ.get("EVOGP_EQ_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

RTOL = 1e-4
ATOL = 1e-4

# Node/function ids as used by the reference (kept locally so the hand-written
# cases below stay independent of reference_eval's own tables).
VAR, CONST, UFUNC, BFUNC, TFUNC = reference.VAR, reference.CONST, reference.UFUNC, reference.BFUNC, reference.TFUNC
OUT_NODE = reference.OUT_NODE
IF, ADD, SUB, MUL, DIV, LOOSE_DIV, POW, LOOSE_POW, MAX, MIN, LT, GT, LE, GE = range(14)
SIN, COS, TAN, SINH, COSH, TANH, LOG, LOOSE_LOG, EXP, INV, LOOSE_INV, NEG, ABS, SQRT, LOOSE_SQRT = range(14, 29)

FUNC_IDS = list(range(29))

# Padding of the unused tail of a row: CONST/NaN. Valid trees may leave the tail
# as uninitialized garbage; NaN is a safe, detectable sentinel (if the engine read
# past subtree_size[0] the result would become NaN while the reference stays finite).
TAIL_TYPE = CONST
TAIL_VALUE = float("nan")


def kind_of(function_id: int) -> int:
    if function_id == IF:
        return TFUNC
    if function_id <= GE:
        return BFUNC
    return UFUNC


def arity_of(function_id: int) -> int:
    return {"tf": 3, "bf": 2, "uf": 1}[{TFUNC: "tf", BFUNC: "bf", UFUNC: "uf"}[kind_of(function_id)]]


# --------------------------------------------------------------------------- #
# Tiny tree-spec DSL: build_prefix arrays for one tree
# --------------------------------------------------------------------------- #
def var(index: int):
    return ("var", index)


def const(value: float):
    return ("const", value)


def fn(function_id: int, *children):
    return ("fn", function_id, children)


def pack_out(function_id: int, out_index: int) -> float:
    """float32 bit pattern of ``struct OutNodeValue { int16_t function, outIndex; }``."""
    bits = np.int32(((int(out_index) & 0xFFFF) << 16) | (int(function_id) & 0xFFFF))
    return float(bits.view(np.float32))


def out_fn(function_id: int, out_index: int, *children):
    return ("out", function_id, out_index, children)


def raw(node_type: int, value: float, *children):
    """Deliberately hand-crafted node (used for out-nodes and invalid-encoding checks)."""
    return ("raw", node_type, float(value), children)


def _type_arity(node_type, multi_output):
    """Children count implied by a node TYPE, exactly as ``_process_node`` selects it.

    In multi-output mode the type is masked with 0x7F first; in single-output mode
    the raw type is compared. Anything unmapped falls into the kernel's final
    ``else`` branch, i.e. it is evaluated as a 3-operand ``IF``.
    """
    t = (node_type & 0x7F) if multi_output else node_type
    if t == UFUNC:
        return 1
    if t == BFUNC:
        return 2
    if t == TFUNC:
        return 3
    if t in (VAR, CONST):
        return 0
    return 3


def _subtree_sizes(types, multi_output=False):
    sizes = [0] * len(types)
    stack = []
    for i in range(len(types) - 1, -1, -1):
        size = 1
        for _ in range(_type_arity(types[i], multi_output)):
            size += stack.pop()
        sizes[i] = size
        stack.append(size)
    assert len(stack) == 1, "spec is not a single valid prefix expression"
    return sizes


def _flat_from_spec(spec, output_len):
    values, types = [], []

    def walk(node):
        tag = node[0]
        if tag == "var":
            values.append(float(node[1]))
            types.append(VAR)
        elif tag == "const":
            values.append(float(node[1]))
            types.append(CONST)
        elif tag == "fn":
            fid, children = node[1], node[2]
            assert len(children) == arity_of(fid), f"function {fid} needs {arity_of(fid)} children"
            values.append(float(fid))
            types.append(kind_of(fid))
            for child in children:
                walk(child)
        elif tag == "out":
            fid, out_index, children = node[1], node[2], node[3]
            assert output_len > 1, "out-nodes are only meaningful for output_len > 1"
            assert len(children) == arity_of(fid)
            values.append(pack_out(fid, out_index))
            types.append(kind_of(fid) | OUT_NODE)
            for child in children:
                walk(child)
        elif tag == "raw":
            node_type, value, children = node[1], node[2], node[3]
            values.append(value)
            types.append(node_type)
            for child in children:
                walk(child)
        else:  # pragma: no cover
            raise ValueError(f"unknown spec tag {tag!r}")

    walk(spec)
    return values, types


def rows_from_specs(specs, output_len=1, pad_to=None, tail_value=TAIL_VALUE):
    """``specs`` -> CPU ``(node_value, node_type, subtree_size)`` tensors (pop, max_tree_len)."""
    flat = [_flat_from_spec(spec, output_len) for spec in specs]
    max_len = pad_to or max(len(v) for v, _ in flat)
    pop = len(flat)
    values = torch.full((pop, max_len), float(tail_value), dtype=torch.float32)
    types = torch.full((pop, max_len), TAIL_TYPE, dtype=torch.int16)
    sizes = torch.zeros((pop, max_len), dtype=torch.int16)
    for i, (v, t) in enumerate(flat):
        assert len(v) <= max_len, f"tree {i} has {len(v)} nodes but pad_to={max_len}"
        values[i, : len(v)] = torch.tensor(v, dtype=torch.float32)
        types[i, : len(v)] = torch.tensor(t, dtype=torch.int16)
        sizes[i, : len(v)] = torch.tensor(_subtree_sizes(t, output_len > 1), dtype=torch.int16)
    return values, types, sizes


def make_forest(specs, input_len, output_len=1, pad_to=None, tail_value=TAIL_VALUE):
    """Build a ``Forest`` from specs (only called from engine-dependent tests)."""
    values, types, sizes = rows_from_specs(specs, output_len, pad_to, tail_value)
    return Forest(
        input_len,
        output_len,
        values.to(DEVICE),
        types.to(DEVICE),
        sizes.to(DEVICE),
    )

# --------------------------------------------------------------------------- #
# Input data
# --------------------------------------------------------------------------- #
def x_random(n, input_len, seed=0, scale=1.0, uniform=False):
    gen = torch.Generator().manual_seed(seed)
    if uniform:
        raw = torch.rand(n, input_len, generator=gen) * 2.0 - 1.0
    else:
        raw = torch.randn(n, input_len, generator=gen)
    return (raw * scale).to(torch.float32)


def adversarial_inputs(input_len=3, n=None, seed=0):
    """(name, x) sets hitting every documented edge case of ``_process_node``."""
    n = N_POINTS if n is None else n
    above_delta = float(np.nextafter(np.float32(1e-9), np.float32(2e-9)))
    below_delta = float(np.nextafter(np.float32(1e-9), np.float32(0.0)))
    cases = [
        ("zeros", torch.zeros(1, input_len)),
        ("signed_zeros", torch.tensor([[-0.0, 0.0, -0.0]] * 1, dtype=torch.float32)),
        ("one_row_of_ones", torch.ones(1, input_len)),
        ("delta_boundary", torch.tensor([[1e-9, -1e-9, 0.0], [above_delta, below_delta, -above_delta]])),
        ("negatives", torch.tensor([[-2.0, -0.5, -1e-30], [-1.0, -1e-9, -3.0]])),
        ("pow_edge", torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0]])),
        ("div_edge", torch.tensor([[0.0, 1.0, -0.0], [1.0, 0.0, 2.0]])),
        ("tiny_huge", torch.tensor([[1e-30, 1e30, -1e30], [1e-8, 1e8, 0.5]])),
        ("random_normal", x_random(max(n, 2), input_len, seed=seed)),
        ("random_uniform", x_random(max(n, 2), input_len, seed=seed + 1, uniform=True)),
        ("random_small", x_random(max(n, 2), input_len, seed=seed + 2, scale=1e-6)),
    ]
    return cases


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def _to_cpu_float(t):
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().to(torch.float32).cpu()


def assert_close_with_placement(actual, expected, *, rtol=RTOL, atol=ATOL, exact=False, context=""):
    """Compare engine output against the oracle, including NaN/±inf PLACEMENT.

    ``torch.testing.assert_close(equal_nan=True)`` cannot detect a missing/extra NaN,
    so the placement of NaN and of ±inf is asserted exactly first; only the finite
    entries are then compared (``rtol``/``atol``, or bit-exact with ``exact=True``).
    """
    got = _to_cpu_float(actual)
    want = _to_cpu_float(expected)
    assert got.shape == want.shape, f"{context}: shape {tuple(got.shape)} != {tuple(want.shape)}"

    got_nan, want_nan = torch.isnan(got), torch.isnan(want)
    assert torch.equal(got_nan, want_nan), (
        f"{context}: NaN placement differs ({int(got_nan.sum())} vs {int(want_nan.sum())} NaNs)\n"
        f"first engine-only NaN index: {_first_mismatch(got_nan, want_nan)}"
    )
    got_inf, want_inf = torch.isinf(got), torch.isinf(want)
    assert torch.equal(got_inf, want_inf), (
        f"{context}: inf placement differs ({int(got_inf.sum())} vs {int(want_inf.sum())} infs)\n"
        f"first engine-only inf index: {_first_mismatch(got_inf, want_inf)}"
    )
    both_inf = got_inf & want_inf
    if both_inf.any():
        assert torch.equal(torch.sign(got[both_inf]), torch.sign(want[both_inf])), (
            f"{context}: inf signs differ"
        )

    finite = ~(got_nan | got_inf)
    a, b = got[finite], want[finite]
    if exact:
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0, msg=lambda m: f"{context}: {m}")
    else:
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=lambda m: f"{context}: {m}")


def _first_mismatch(a, b):
    diff = (a != b).reshape(-1)
    index = int(torch.nonzero(diff, as_tuple=False)[0]) if diff.any() else -1
    return index


def engine_batch_forward(forest, x):
    return forest.batch_forward(x.to(dtype=torch.float32, device=DEVICE))


def reference_batch_forward(forest, x):
    return reference.batch_forward_from_rows(
        forest.batch_node_value.detach().cpu().float(),
        forest.batch_node_type.detach().cpu(),
        forest.batch_subtree_size.detach().cpu(),
        x.detach().cpu().float(),
        out_len=forest.output_len,
    )


def reference_sr_fitness(forest, inputs, labels, use_mse=True):
    return reference.sr_fitness_from_rows(
        forest.batch_node_value.detach().cpu().float(),
        forest.batch_node_type.detach().cpu(),
        forest.batch_subtree_size.detach().cpu(),
        inputs.detach().cpu().float(),
        labels.detach().cpu().float(),
        use_mse=use_mse,
        out_len=forest.output_len,
    )


def compare_batch_forward(forest, x, context="", exact=False):
    got = engine_batch_forward(forest, x)
    want = reference_batch_forward(forest, x)
    assert_close_with_placement(got, want, exact=exact, context=context)
    return got


# --------------------------------------------------------------------------- #
# Populations
# --------------------------------------------------------------------------- #
def seed_all(seed):
    """Seed the global torch RNGs (CPU and every CUDA device) reproducibly.

    ``GenerateDescriptor`` draws its ``const_samples`` from the global CUDA RNG
    and ``Forest.random_generate`` draws the population from the global RNG of
    the target device, so both must be seeded before either happens.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_random_population(pop_size, seed, **descriptor_kwargs):
    """Deterministic ``(descriptor, forest)`` pair: seed -> descriptor -> population.

    The descriptor is built *after* seeding so its ``const_samples`` draw is part
    of the reproducible state; two calls with the same ``seed``/``kwargs`` give
    bit-identical populations (and therefore bit-identical engine results).
    """
    seed_all(seed)
    descriptor = make_descriptor(**descriptor_kwargs)
    forest = Forest.random_generate(pop_size=pop_size, descriptor=descriptor)
    return descriptor, forest


def make_descriptor(max_tree_len=64, input_len=3, output_len=1, using_funcs=None, **kwargs):
    using_funcs = using_funcs or ["+", "-", "*", "/"]
    kwargs.setdefault("max_layer_cnt", 4)
    kwargs.setdefault("layer_leaf_prob", 0.2)
    kwargs.setdefault("const_range", (-2.0, 2.0))
    kwargs.setdefault("sample_cnt", 8)
    return GenerateDescriptor(
        max_tree_len=max_tree_len,
        input_len=input_len,
        output_len=output_len,
        using_funcs=using_funcs,
        **kwargs,
    )


# Every function id 0..28 once, on hand-built rows (input_len >= 3).
def curated_function_population():
    specs = []
    for fid in FUNC_IDS:
        if fid == IF:
            specs.append(fn(fid, var(0), var(1), var(2)))
        elif kind_of(fid) == BFUNC:
            specs.append(fn(fid, var(0), var(1)))
        else:
            specs.append(fn(fid, var(0)))
    return specs


def curated_edge_population():
    """Adversarial hand-built trees: div-by-0, ±1e-9, log/sqrt of 0/neg, 0^0, IF."""
    nan_op = fn(DIV, var(1), const(0.0))
    return [
        const(3.5),
        var(2),
        fn(DIV, var(1), const(0.0)),
        fn(DIV, const(1.0), const(-0.0)),
        fn(LOOSE_DIV, var(1), const(0.0)),
        fn(LOOSE_DIV, const(1.0), const(-1e-12)),
        fn(LOOSE_DIV, const(1.0), const(1e-12)),
        fn(LOOSE_POW, const(0.0), const(0.0)),
        fn(LOOSE_POW, const(-2.0), const(3.0)),
        fn(POW, const(-2.0), const(0.5)),
        fn(LOG, const(0.0)),
        fn(LOG, const(-1.0)),
        fn(LOOSE_LOG, const(0.0)),
        fn(LOOSE_LOG, const(-0.5)),
        fn(INV, const(0.0)),
        fn(INV, const(-0.0)),
        fn(LOOSE_INV, const(0.0)),
        fn(LOOSE_INV, const(-0.0)),
        fn(SQRT, const(-1.0)),
        fn(LOOSE_SQRT, const(-4.0)),
        fn(IF, const(1.0), var(1), var(2)),
        fn(IF, const(0.0), var(1), var(2)),
        fn(IF, const(-1.0), var(1), var(2)),
        fn(IF, nan_op, var(1), var(2)),
        fn(EXP, const(1e30)),
        fn(EXP, const(-1e30)),
        fn(SQRT, const(0.0)),
        fn(ABS, nan_op),
        fn(NEG, nan_op),
        # nested: NaN flows through a comparison (always -1) and through ADD
        fn(ADD, nan_op, const(1.0)),
        fn(LT, nan_op, var(0)),
        fn(MAX, nan_op, const(1.0)),
        # the sign of zero must survive sinh: loose_inv(-0.0) -> 1 / -1e-9 = -1e9
        fn(LOOSE_INV, fn(SINH, const(-0.0))),
        fn(LOOSE_INV, fn(SINH, const(0.0))),
        fn(EXP, fn(SINH, const(-0.0))),
        # ... on the SYMBOLIC path too (a constant operand is folded on the host):
        # x0 = +0.0 -> neg -> -0.0 -> sinh -> -0.0 -> 1 / -1e-9 = -1e9
        fn(LOOSE_INV, fn(SINH, fn(NEG, var(0)))),
        fn(LOOSE_INV, fn(SINH, var(0))),
        fn(EXP, fn(SINH, fn(NEG, var(0)))),
        fn(LOOSE_DIV, const(1.0), fn(SINH, fn(NEG, var(0)))),
    ]


def deep_skewed_population(size=1000, max_tree_len=1024):
    """Deep left/right chains, all-constant subtrees and a MAX_STACK-sized tree."""
    left = var(0)
    for i in range(size // 2):
        left = fn(ADD, left, const(1.0))
    right = var(1)
    for i in range(size // 2):
        right = fn(MUL, const(1.0), right)
    return [
        left,
        right,
        fn(MUL, fn(ADD, const(1.0), const(2.0)), var(1)),  # all-constant subtree
        fn(ADD, var(0), const(0.0)) if size > 2 else const(0.0),
        const(1.0),
    ]


def a_max_stack_tree():
    """A single 1023-node left-skewed chain (MAX_STACK = 1024 bound)."""
    node = var(0)
    for _ in range(511):
        node = fn(ADD, node, const(1.0))
    return node


def linear_chain(nodes):
    """Left-skewed ADD chain with ``nodes`` nodes (odd, >= 1)."""
    node = var(0)
    for _ in range((nodes - 1) // 2):
        node = fn(ADD, node, const(1.0))
    return node


def engine_sr_fitness(forest, inputs, labels, use_mse=True, execute_mode="auto"):
    """Single call site for the documented ``SR_fitness(inputs, labels, use_MSE, execute_mode)``."""
    return forest.SR_fitness(inputs, labels, use_mse, execute_mode)


def rows_of_tree(tree):
    """1-row ``(node_value, node_type, subtree_size)`` CPU tensors of a ``Tree``."""
    return (
        tree.node_value.detach().cpu().float().unsqueeze(0),
        tree.node_type.detach().cpu().unsqueeze(0),
        tree.subtree_size.detach().cpu().unsqueeze(0),
    )


def _normalized(t):
    t = _to_cpu_float(t)
    return torch.nan_to_num(t, nan=12345.0, posinf=1e31, neginf=-1e31)


def assert_same_bits(actual, expected, context=""):
    """Bit-level equality up to a NaN/±inf canonicalization (NaN != NaN otherwise)."""
    got, want = _normalized(actual), _normalized(expected)
    assert got.shape == want.shape, f"{context}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    assert torch.equal(got, want), f"{context}: values differ (max |diff| = {float((got - want).abs().max())})"


# =========================================================================== #
# 1. Oracle calibration -- hand-computed values taken from cuda/forward.cu
#    (these run even when the engine is absent, they validate the reference)
# =========================================================================== #
_X_HAND = torch.tensor([[0.0, 2.0, 4.0]], dtype=torch.float32)  # x0=0, x1=2, x2=4
_NAN = float("nan")

_HAND_CASES = [
    # (name, spec, expected, exact)
    ("var", var(1), 2.0, True),
    ("const", const(2.5), 2.5, True),
    ("const_nan", const(_NAN), _NAN, True),
    ("const_pos_inf", const(float("inf")), float("inf"), True),
    ("add", fn(ADD, var(1), var(2)), 6.0, True),
    ("sub_leftmost_minus_rightmost", fn(SUB, var(1), var(2)), -2.0, True),
    ("sub_reversed", fn(SUB, var(2), var(1)), 2.0, True),
    ("mul", fn(MUL, var(1), var(2)), 8.0, True),
    ("div", fn(DIV, var(2), var(1)), 2.0, True),
    ("div_by_zero_is_nan", fn(DIV, var(1), var(0)), _NAN, True),
    ("div_by_neg_zero_is_nan", fn(DIV, const(1.0), const(-0.0)), _NAN, True),
    ("loose_div_zero", fn(LOOSE_DIV, var(1), var(0)), 2e9, False),
    ("loose_div_neg_tiny", fn(LOOSE_DIV, const(1.0), const(-1e-12)), -1e9, False),
    ("loose_div_tiny_positive", fn(LOOSE_DIV, const(1.0), const(1e-12)), 1e9, False),
    ("pow", fn(POW, var(1), var(2)), 16.0, True),
    ("pow_neg_frac_is_nan", fn(POW, const(-2.0), const(0.5)), _NAN, True),
    ("loose_pow_zero_zero", fn(LOOSE_POW, var(0), var(0)), 0.0, True),
    ("loose_pow_neg_base_odd_exp", fn(LOOSE_POW, const(-2.0), const(3.0)), 8.0, True),
    ("loose_pow_neg_base_even_exp", fn(LOOSE_POW, const(-2.0), const(2.0)), 4.0, True),
    ("max", fn(MAX, var(1), var(2)), 4.0, True),
    ("min", fn(MIN, var(1), var(2)), 2.0, True),
    ("max_with_zero", fn(MAX, var(0), const(-1.0)), 0.0, True),
    ("lt", fn(LT, var(1), var(2)), 1.0, True),
    ("gt", fn(GT, var(1), var(2)), -1.0, True),
    ("le", fn(LE, var(1), var(2)), 1.0, True),
    ("ge", fn(GE, var(1), var(2)), -1.0, True),
    ("lt_equal_is_minus_one", fn(LT, var(1), var(1)), -1.0, True),
    ("gt_equal_is_minus_one", fn(GT, var(1), var(1)), -1.0, True),
    ("le_equal_is_one", fn(LE, var(1), var(1)), 1.0, True),
    ("ge_equal_is_one", fn(GE, var(1), var(1)), 1.0, True),
    ("lt_nan_operand_is_minus_one", fn(LT, fn(DIV, var(1), const(0.0)), var(1)), -1.0, True),
    ("sin_zero", fn(SIN, var(0)), 0.0, True),
    ("cos_zero", fn(COS, var(0)), 1.0, True),
    ("tan_zero", fn(TAN, var(0)), 0.0, True),
    ("sinh_zero", fn(SINH, var(0)), 0.0, True),
    ("cosh_zero", fn(COSH, var(0)), 1.0, True),
    ("tanh_zero", fn(TANH, var(0)), 0.0, True),
    ("exp_zero", fn(EXP, var(0)), 1.0, True),
    ("log_zero_is_neg_inf", fn(LOG, var(0)), float("-inf"), True),
    ("log_negative_is_nan", fn(LOG, const(-1.0)), _NAN, True),
    ("log_two", fn(LOG, var(1)), math.log(2.0), False),
    ("loose_log_zero", fn(LOOSE_LOG, var(0)), -1e9, True),
    ("loose_log_negative", fn(LOOSE_LOG, const(-0.5)), -math.log(2.0), False),
    ("loose_log_one", fn(LOOSE_LOG, const(1.0)), 0.0, True),
    ("inv_zero_is_nan", fn(INV, var(0)), _NAN, True),
    ("inv", fn(INV, const(-2.0)), -0.5, True),
    ("loose_inv_zero", fn(LOOSE_INV, var(0)), 1e9, False),
    ("loose_inv_neg_zero", fn(LOOSE_INV, const(-0.0)), -1e9, False),
    ("neg", fn(NEG, var(1)), -2.0, True),
    ("abs_negative", fn(ABS, const(-3.0)), 3.0, True),
    ("sqrt", fn(SQRT, const(4.0)), 2.0, True),
    ("sqrt_negative_is_nan", fn(SQRT, const(-1.0)), _NAN, True),
    ("loose_sqrt_negative", fn(LOOSE_SQRT, const(-4.0)), 2.0, True),
    ("loose_sqrt_zero", fn(LOOSE_SQRT, var(0)), 0.0, True),
    ("if_true", fn(IF, const(1.0), var(1), var(2)), 2.0, True),
    ("if_zero_is_false", fn(IF, const(0.0), var(1), var(2)), 4.0, True),
    ("if_negative_is_false", fn(IF, const(-1.0), var(1), var(2)), 4.0, True),
    # encoding details the kernels define but generation never produces
    ("tfunc_ignores_function_id", raw(TFUNC, float(ADD), var(0), var(1), var(2)), 4.0, True),
    ("unmapped_type_is_tfunc", raw(5, 0.0, var(0), var(1), var(2)), 4.0, True),
    ("out_flagged_type_in_single_output_mode_is_tfunc", raw(UFUNC | OUT_NODE, 0.0, var(0), var(1), var(2)), 4.0, True),
    ("bfunc_with_unary_id_is_zero", raw(BFUNC, float(SIN), var(0), var(1)), 0.0, True),
    ("bfunc_with_unknown_id_is_zero", raw(BFUNC, 30.0, var(0), var(1)), 0.0, True),
    ("ufunc_with_binary_id_is_zero", raw(UFUNC, float(ADD), var(0)), 0.0, True),
]


@pytest.mark.parametrize("name,spec,expected,exact", _HAND_CASES, ids=[case[0] for case in _HAND_CASES])
def test_reference_hand_computed_values(name, spec, expected, exact):
    values, types, sizes = rows_from_specs([spec])
    got = reference.forward_from_rows(values, types, sizes, _X_HAND)
    assert tuple(got.shape) == (1, 1), f"{name}: expected (1, 1), got {tuple(got.shape)}"
    value = float(got[0, 0])
    if math.isnan(expected):
        assert math.isnan(value), f"{name}: expected NaN, got {value}"
    elif math.isinf(expected):
        assert value == expected, f"{name}: expected {expected}, got {value}"
    elif exact:
        assert value == expected, f"{name}: expected {expected}, got {value}"
    else:
        assert value == pytest.approx(expected, rel=1e-5, abs=1e-12), f"{name}: expected {expected}, got {value}"


def test_reference_operand_order_and_single_row_scores():
    """SR fitness = sum over points of sum over outputs of err / data_points (positive)."""
    inputs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    values, types, sizes = rows_from_specs([fn(ADD, var(0), const(1.0))])  # pred = 2, 3, 4
    mse = reference.sr_fitness_from_rows(values, types, sizes, inputs, labels, use_mse=True)
    mae = reference.sr_fitness_from_rows(values, types, sizes, inputs, labels, use_mse=False)
    assert tuple(mse.shape) == (1,)
    assert float(mse[0]) == pytest.approx(14.0 / 3.0, rel=1e-6)  # (1+4+9)/3
    assert float(mae[0]) == pytest.approx(2.0, rel=1e-6)  # (1+2+3)/3
    # the loss is NOT negated at this layer and must be non-negative
    assert float(mse[0]) > 0.0


def test_reference_multi_output_accumulation():
    """Out-nodes accumulate into out[outIdx]; the right-most operand is passed up."""
    inputs = torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=torch.float32)
    spec = fn(ADD, out_fn(ADD, 0, var(0), var(1)), out_fn(MUL, 1, var(1), var(2)))
    values, types, sizes = rows_from_specs([spec], output_len=2)
    out = reference.batch_forward_from_rows(values, types, sizes, inputs, out_len=2)
    expected = torch.tensor(
        [[[2.0, 8.0], [2.0, 1.0], [5.0, 12.0]]], dtype=torch.float32
    )  # out0 = x0 + x1, out1 = x1 * x2
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # two out-nodes writing the same index sum up; the second output stays 0.
    # Because every node passes its RIGHT-MOST operand up, the outer ADD sees
    # (inner_rightmost=var(1)=2, const(3.0)) and the root sees (const(3.0), const(0.0)):
    # inner out-node (0 + 2) -> out[0] += 2 ; root (3 + 0) -> out[0] += 3.
    spec_shared = fn(ADD, out_fn(ADD, 0, var(0), var(1)), const(3.0))
    spec_nested = out_fn(ADD, 0, spec_shared, const(0.0))
    values, types, sizes = rows_from_specs([spec_nested], output_len=2)
    out = reference.batch_forward_from_rows(values, types, sizes, inputs[:1], out_len=2)
    torch.testing.assert_close(out[0, 0], torch.tensor([5.0, 0.0]), rtol=0, atol=0)

    # out indices >= out_len are dropped (forward.cu: `outIdx < outLen`)
    values, types, sizes = rows_from_specs(
        [out_fn(ADD, 3, var(0), var(1))], output_len=2, pad_to=3
    )
    out = reference.batch_forward_from_rows(values, types, sizes, inputs[:1], out_len=2)
    torch.testing.assert_close(out[0, 0], torch.tensor([0.0, 0.0]), rtol=0, atol=0)


def test_reference_multi_output_fitness_divisor():
    """Outputs are summed but the divisor is data_points only (never out_len)."""
    inputs = torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=torch.float32)
    labels = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    spec = fn(ADD, out_fn(ADD, 0, var(0), var(1)), out_fn(MUL, 1, var(1), var(2)))
    values, types, sizes = rows_from_specs([spec], output_len=2)
    mse = reference.sr_fitness_from_rows(values, types, sizes, inputs, labels, use_mse=True)
    mae = reference.sr_fitness_from_rows(values, types, sizes, inputs, labels, use_mse=False)
    # preds: [2,8] [2,1] [5,12]; errs: 68, 1, 169 -> 238/3
    assert float(mse[0]) == pytest.approx(238.0 / 3.0, rel=1e-6)
    assert float(mae[0]) == pytest.approx(28.0 / 3.0, rel=1e-6)


def test_reference_rejects_inconsistent_encoding():
    values, types, sizes = rows_from_specs([fn(ADD, var(0), var(1))])
    good = (values[0], types[0], sizes[0])
    # declared subtree size larger than the row
    with pytest.raises(ValueError):
        reference.decode_row(good[0], good[1], torch.tensor([99, 1, 1, 1], dtype=torch.int16))
    # prefix consumed more nodes than subtree_size[0]
    with pytest.raises(ValueError):
        reference.decode_row(good[0], good[1], torch.tensor([2, 1, 1, 1], dtype=torch.int16))
    # leftover sub-trees (an extra leaf the root cannot consume)
    bad_types = torch.tensor([BFUNC, VAR, VAR, VAR], dtype=torch.int16)
    bad_values = torch.tensor([float(ADD), 0.0, 1.0, 2.0], dtype=torch.float32)
    bad_sizes = torch.tensor([4, 1, 1, 1], dtype=torch.int16)
    with pytest.raises(ValueError):
        reference.decode_row(bad_values, bad_types, bad_sizes)
    # declared per-node size inconsistent with the tree
    with pytest.raises(ValueError):
        reference.decode_row(good[0], good[1], torch.tensor([3, 2, 1, 1], dtype=torch.int16))
    # VAR index outside input_len must not silently read out of bounds
    values, types, sizes = rows_from_specs([var(7)])
    with pytest.raises(IndexError):
        reference.forward_tree(reference.decode_row(values[0], types[0], sizes[0]), _X_HAND)


# --------------------------------------------------------------------------- #
# Correctly rounded float32 calibration of the oracle
# --------------------------------------------------------------------------- #
# The oracle must give the best possible float32 answer at every node: the
# transcendentals are evaluated on float64 operands and rounded back to float32,
# the hardware-exact ops stay float32 (identical by IEEE 754). These cases check
# the oracle bit-for-bit against a numpy float64 reference of the same
# documented semantics, and they are what makes the engine-vs-oracle comparison
# meaningful: without them a one-ulp oracle error would look like an engine bug.
_CR_SAMPLES = np.asarray(
    [
        0.0, -0.0, 1e-30, -1e-30, 1e-9, -1e-9,
        float(np.nextafter(np.float32(1e-4), np.float32(0.0))), 1e-4,
        0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.5, 7.3, -7.3, 0.12345, -0.98765,
        16.0, 100.0, -100.0, 1e9, -1e9, 1e30, -1e30,
    ],
    dtype=np.float32,
)

_CR_UNARY_CASES = [
    ("sin", SIN, np.sin),
    ("cos", COS, np.cos),
    ("tan", TAN, np.tan),
    ("sinh", SINH, np.sinh),
    ("cosh", COSH, np.cosh),
    ("tanh", TANH, np.tanh),
    ("exp", EXP, np.exp),
    ("log", LOG, np.log),
]

_CR_PAIRS = [
    (0.0, 0.0), (-0.0, 0.0), (0.0, -1.0), (2.0, 3.0), (-2.0, 3.0), (2.0, 0.5),
    (-2.0, 0.5), (2.0, -3.0), (0.5, 1e9), (1e30, 2.0), (-1e30, 3.0), (7.3, -1.5),
    (1e-30, -2.0), (3.5, 0.0), (-3.5, 0.0),
]
# the operands travel through float32 node values, so the float64 reference must
# start from their float32 rounding (7.3 != float64(7.3))
_CR_PAIR_A = np.asarray([a for a, _ in _CR_PAIRS], dtype=np.float32)
_CR_PAIR_B = np.asarray([b for _, b in _CR_PAIRS], dtype=np.float32)


def _cr64(result) -> torch.Tensor:
    """Round a numpy float64 result to float32 (the node-value storage)."""
    return torch.from_numpy(np.asarray(result, dtype=np.float32))


def _f64_col(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@pytest.mark.parametrize("name,function_id,np_op", _CR_UNARY_CASES, ids=[c[0] for c in _CR_UNARY_CASES])
def test_reference_transcendentals_are_correctly_rounded(name, function_id, np_op):
    specs = [fn(function_id, const(float(v))) for v in _CR_SAMPLES]
    values, types, sizes = rows_from_specs(specs)
    got = reference.batch_forward_from_rows(values, types, sizes, _X_HAND)
    with np.errstate(all="ignore"):
        want = _cr64(np_op(_f64_col(_CR_SAMPLES)))
    assert_same_bits(got[:, 0, 0], want, context=f"oracle {name} must be correctly rounded")


def test_reference_loose_log_is_correctly_rounded():
    """``loose_log(+-0) = -MAX_VAL`` (engine spec), everything else ``log|a|``."""
    specs = [fn(LOOSE_LOG, const(float(v))) for v in _CR_SAMPLES]
    values, types, sizes = rows_from_specs(specs)
    got = reference.batch_forward_from_rows(values, types, sizes, _X_HAND)
    with np.errstate(all="ignore"):
        want = np.asarray(np.log(np.abs(_f64_col(_CR_SAMPLES))), dtype=np.float32)
    want[_CR_SAMPLES == 0.0] = np.float32(-float(reference.MAX_VAL))
    assert_same_bits(got[:, 0, 0], torch.from_numpy(want), context="oracle loose_log must be correctly rounded")


def test_reference_pow_is_correctly_rounded():
    specs = [fn(POW, const(a), const(b)) for a, b in _CR_PAIRS]
    values, types, sizes = rows_from_specs(specs)
    got = reference.batch_forward_from_rows(values, types, sizes, _X_HAND)
    with np.errstate(all="ignore"):
        want = _cr64(np.power(_f64_col(_CR_PAIR_A), _f64_col(_CR_PAIR_B)))
    assert_same_bits(got[:, 0, 0], want, context="oracle pow must be correctly rounded")


def test_reference_loose_pow_is_correctly_rounded():
    """``loose_pow(0, 0) = 0`` (engine spec), otherwise ``|a| ** b``."""
    specs = [fn(LOOSE_POW, const(a), const(b)) for a, b in _CR_PAIRS]
    values, types, sizes = rows_from_specs(specs)
    got = reference.batch_forward_from_rows(values, types, sizes, _X_HAND)
    with np.errstate(all="ignore"):
        want = np.asarray(
            np.power(np.abs(_f64_col(_CR_PAIR_A)), _f64_col(_CR_PAIR_B)), dtype=np.float32
        )
    for i, (a, b) in enumerate(_CR_PAIRS):
        if a == 0.0 and b == 0.0:
            want[i] = np.float32(0.0)
    assert_same_bits(got[:, 0, 0], torch.from_numpy(want), context="oracle loose_pow must be correctly rounded")


def test_reference_sinh_keeps_tiny_values_and_signed_zero():
    """Correctly rounded ``sinh``: exactly ``x`` for ``|x| < 1e-4`` and ``sinh(-0.0) = -0.0``."""
    tiny = float(np.nextafter(np.float32(1e-4), np.float32(0.0)))
    samples = [0.0, -0.0, 1e-30, -1e-30, 1e-9, -1e-9, tiny]
    specs = [fn(SINH, const(v)) for v in samples]
    values, types, sizes = rows_from_specs(specs)
    got = reference.batch_forward_from_rows(values, types, sizes, _X_HAND)[:, 0, 0]
    want = torch.tensor(samples, dtype=torch.float32)
    assert torch.equal(got, want), f"sinh(|x| < 1e-4) must be exactly x, got {got.tolist()}"
    assert bool(torch.signbit(got[1])), "sinh(-0.0) must stay -0.0"
    # and it is the correctly rounded value of the true sinh(float64(x)) as well
    assert_same_bits(
        got, _cr64(np.sinh(_f64_col(want))), context="oracle sinh must be correctly rounded"
    )
# =========================================================================== #
# 2. Public API surface / small unit checks
# =========================================================================== #
@requires_engine
def test_public_api_symbols():
    assert MAX_STACK == 1024, f"MAX_STACK changed: {MAX_STACK}"
    assert NType.VAR == VAR and NType.CONST == CONST
    assert NType.UFUNC == UFUNC and NType.BFUNC == BFUNC and NType.TFUNC == TFUNC
    assert NType.OUT_NODE == OUT_NODE and NType.TYPE_MASK == 0x7F
    indices = randint(8, 0, 3)
    assert tuple(indices.shape) == (8,)
    assert indices.dtype == torch.int32
    assert 0 <= int(indices.min()) and int(indices.max()) < 3


# =========================================================================== #
# 3. Randomly generated populations through the public API
# =========================================================================== #
@requires_engine
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_random_population_vs_reference(seed):
    """Default-scale random population built by Forest.random_generate, all 29 functions."""
    all_funcs = [
        "if", "+", "-", "*", "/", "loose_div", "pow", "loose_pow", "max", "min",
        "<", ">", "<=", ">=", "sin", "cos", "tan", "sinh", "cosh", "tanh",
        "log", "loose_log", "exp", "inv", "loose_inv", "neg", "abs", "sqrt", "loose_sqrt",
    ]
    _, forest = make_random_population(TREES, seed, max_tree_len=64, input_len=3, using_funcs=all_funcs)
    assert forest.pop_size == TREES
    for name, x in [
        ("normal", x_random(N_POINTS, 3, seed=SEED + 7)),
        ("uniform", x_random(N_POINTS, 3, seed=SEED + 8, uniform=True)),
        ("zeros", torch.zeros(N_POINTS, 3)),
    ]:
        compare_batch_forward(forest, x, context=f"random population seed={seed} / {name}")


_DESCRIPTOR_VARIANTS = [
    ("arithmetic", dict(max_tree_len=32, max_layer_cnt=3, using_funcs=["+", "-", "*", "/"])),
    ("unary", dict(max_tree_len=64, max_layer_cnt=5, using_funcs=["sin", "cos", "exp", "neg", "abs", "sqrt"])),
    ("if_heavy", dict(max_tree_len=80, max_layer_cnt=4, using_funcs=["if", "+", "-"], const_prob=0.8)),
    ("loose_ops", dict(max_tree_len=40, max_layer_cnt=3, using_funcs=["loose_div", "loose_pow", "loose_log", "loose_inv", "loose_sqrt"])),
    ("comparisons", dict(max_tree_len=64, max_layer_cnt=4, using_funcs=["<", ">", "<=", ">=", "+", "-"])),
    ("no_constants", dict(max_tree_len=64, max_layer_cnt=4, using_funcs=["+", "-", "*", "/"], const_prob=0.0)),
    ("all_constants", dict(max_tree_len=64, max_layer_cnt=4, using_funcs=["+", "-", "*", "/"], const_prob=1.0)),
    ("deep_zero_leaf_prob", dict(max_tree_len=64, max_layer_cnt=10, using_funcs=["sin", "neg"], layer_leaf_prob=0.0)),
    ("shallow", dict(max_tree_len=64, max_layer_cnt=5, using_funcs=["+", "-", "*", "/"], layer_leaf_prob=0.9)),
    (
        "mixed",
        dict(
            max_tree_len=64,
            max_layer_cnt=4,
            using_funcs=["+", "-", "*", "/", "if", "max", "min", "pow", "tanh", "log", "inv", "loose_div"],
        ),
    ),
]


@requires_engine
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
@pytest.mark.parametrize("name,kwargs", _DESCRIPTOR_VARIANTS, ids=[v[0] for v in _DESCRIPTOR_VARIANTS])
def test_random_descriptors_vs_reference(name, kwargs, seed):
    pop_size = max(TREES // 10, 50)
    descriptor, forest = make_random_population(pop_size, seed, **kwargs)
    for x_name, x in [
        ("normal", x_random(N_POINTS, descriptor.input_len, seed=SEED + 13)),
        ("ones", torch.ones(3, descriptor.input_len)),
    ]:
        compare_batch_forward(forest, x, context=f"descriptor {name} seed={seed} / {x_name}")


# =========================================================================== #
# 4. Hand-built (= adversarial) populations
# =========================================================================== #
@requires_engine
def test_curated_every_function_id_vs_reference():
    """One tree per function id 0..28 on every adversarial input set."""
    specs = curated_function_population()
    assert len(specs) == 29
    forest = make_forest(specs, input_len=3)
    assert forest.pop_size == 29
    for name, x in adversarial_inputs(input_len=3):
        compare_batch_forward(forest, x, context=f"every function id / {name}")


@requires_engine
def test_adversarial_trees_vs_reference():
    """div-by-0, ±1e-9/±1e-12 boundaries, log/sqrt of 0 and negatives, 0^0, IF, overflow."""
    forest = make_forest(curated_edge_population(), input_len=3)
    for name, x in adversarial_inputs(input_len=3):
        compare_batch_forward(forest, x, context=f"adversarial trees / {name}")
    # single data point / N=1 shapes
    compare_batch_forward(forest, torch.tensor([[0.0, 2.0, 4.0]]), context="adversarial trees / N=1")
    compare_batch_forward(forest, torch.tensor([[-1e-9, 1e-9, 0.0]]), context="adversarial trees / N=1 delta")


@requires_engine
def test_nan_inf_constants_vs_reference():
    specs = [
        const(float("nan")),
        const(float("inf")),
        const(float("-inf")),
        fn(ADD, const(float("inf")), const(1.0)),
        fn(ADD, const(float("nan")), const(1.0)),
        fn(MUL, const(0.0), const(float("inf"))),
        fn(IF, const(float("nan")), const(1.0), const(2.0)),
        fn(MAX, const(float("nan")), const(1.0)),
        fn(MIN, const(float("nan")), const(1.0)),
        fn(LT, const(float("inf")), const(float("inf"))),
    ]
    forest = make_forest(specs, input_len=3)
    for name, x in adversarial_inputs(input_len=3, n=4):
        compare_batch_forward(forest, x, context=f"nan/inf constants / {name}")


@requires_engine
def test_comparison_operators_are_exact():
    """LT/GT/LE/GE must be bit-exact (±1): only exactly-representable operands are used."""
    operands = [
        var(0),
        var(1),
        const(0.0),
        const(-0.0),
        const(1.0),
        const(-1.0),
        const(2.5),
        fn(ADD, var(0), var(1)),
        fn(SUB, var(0), var(1)),
        fn(MUL, var(0), const(2.0)),
        fn(NEG, var(1)),
        fn(ABS, var(2)),
        fn(DIV, var(1), const(0.0)),  # exact NaN in both implementations
        fn(INV, const(0.0)),  # exact NaN
    ]
    specs = [fn(op, a, b) for op in (LT, GT, LE, GE) for a in operands for b in operands]
    forest = make_forest(specs, input_len=3)
    exact_x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0], [-1e-9, 1e-9, 0.0]], dtype=torch.float32)
    for name, x in [("exact_values", exact_x)] + adversarial_inputs(input_len=3, n=16):
        compare_batch_forward(forest, x, context=f"comparison operators / {name}", exact=True)


@requires_engine
def test_deep_skewed_and_constant_subtrees_vs_reference():
    specs = deep_skewed_population(size=200) + [a_max_stack_tree(), linear_chain(63), linear_chain(1023)]
    forest = make_forest(specs, input_len=3, pad_to=MAX_STACK)
    compare_batch_forward(forest, x_random(4, 3, seed=SEED + 31), context="deep/skewed trees")
    compare_batch_forward(forest, torch.zeros(1, 3), context="deep/skewed trees / zeros")
    # all-constant subtrees / single-node trees
    constant_specs = [
        const(0.0),
        const(-1e-9),
        const(1e30),
        fn(ADD, fn(MUL, const(2.0), const(3.0)), fn(SUB, const(5.0), const(1.0))),
        fn(MUL, fn(ADD, const(1.0), const(2.0)), var(1)),
        fn(DIV, fn(ADD, const(1.0), const(2.0)), const(0.0)),
    ]
    constant_forest = make_forest(constant_specs, input_len=3)
    for name, x in adversarial_inputs(input_len=3, n=8):
        compare_batch_forward(constant_forest, x, context=f"constant subtrees / {name}")


# =========================================================================== #
# 5. forward / SR_fitness entry points
# =========================================================================== #
@requires_engine
def test_forest_forward_pop_vs_reference():
    """Forest.forward: (pop_size, input_len) -> (pop_size, output_len)."""
    for specs, input_len, out_len, tag in [
        (curated_edge_population(), 3, 1, "curated"),
        (curated_function_population(), 3, 1, "all-ids"),
    ]:
        forest = make_forest(specs, input_len=input_len, output_len=out_len)
        x = x_random(forest.pop_size, input_len, seed=SEED + 41)
        got = forest.forward(x.to(DEVICE))
        assert tuple(got.shape) == (forest.pop_size, out_len)
        want = reference.forward_from_rows(
            forest.batch_node_value.detach().cpu().float(),
            forest.batch_node_type.detach().cpu(),
            forest.batch_subtree_size.detach().cpu(),
            x,
            out_len=out_len,
        )
        assert_close_with_placement(got, want, context=f"Forest.forward / {tag}")


@requires_engine
def test_tree_forward_vs_reference():
    input_len = 3
    forest = make_forest(curated_edge_population(), input_len=input_len)
    tree = forest[5]
    assert isinstance(tree, Tree)

    row = x_random(1, input_len, seed=SEED + 51)[0]
    got_1d = tree.forward(row.to(DEVICE))
    assert tuple(got_1d.shape) == (1,), f"Tree.forward(1-D) should give (output_len,), got {tuple(got_1d.shape)}"

    batch = x_random(7, input_len, seed=SEED + 52)
    got_2d = tree.forward(batch.to(DEVICE))
    assert tuple(got_2d.shape) == (7, 1)

    values, types, sizes = rows_of_tree(tree)
    want_1d = reference.forward_from_rows(values, types, sizes, row.unsqueeze(0))
    want_2d = reference.batch_forward_from_rows(values, types, sizes, batch)
    assert_close_with_placement(got_1d, want_1d[0], context="Tree.forward(1-D)")
    assert_close_with_placement(got_2d, want_2d[0], context="Tree.forward(2-D)")

    # a batch of one must equal the single-tree call (same rows, same engine path)
    single_forest = Forest(
        input_len,
        1,
        tree.node_value.detach().cpu().float().unsqueeze(0).to(DEVICE),
        tree.node_type.detach().cpu().unsqueeze(0).to(DEVICE),
        tree.subtree_size.detach().cpu().unsqueeze(0).to(DEVICE),
    )
    assert_close_with_placement(
        single_forest.forward(row.unsqueeze(0).to(DEVICE))[0], got_1d, context="Forest(1).forward vs Tree.forward"
    )


@requires_engine
def test_sr_fitness_vs_reference():
    """SR_fitness: (pop,) positive loss, MSE and MAE, several data sizes."""
    input_len = 3
    populations = [
        ("curated_edge", make_forest(curated_edge_population(), input_len=input_len)),
        ("all_ids", make_forest(curated_function_population(), input_len=input_len)),
        (
            "random",
            make_random_population(min(TREES, 300), SEED, max_tree_len=64, input_len=input_len)[1],
        ),
    ]
    for tag, forest in populations:
        for n_points in (1, 2, 37, N_POINTS):
            inputs = x_random(n_points, input_len, seed=SEED + 61)
            labels = x_random(n_points, forest.output_len, seed=SEED + 62)
            for use_mse in (True, False):
                context = f"SR_fitness {tag} / N={n_points} / mse={use_mse}"
                got = engine_sr_fitness(forest, inputs.to(DEVICE), labels.to(DEVICE), use_mse)
                assert tuple(got.shape) == (forest.pop_size,), f"{context}: bad shape {tuple(got.shape)}"
                want = reference_sr_fitness(forest, inputs, labels, use_mse)
                assert_close_with_placement(got, want, context=context)
    # perfect fit -> exactly zero loss
    forest = make_forest([fn(ADD, var(0), var(1))], input_len=input_len)
    inputs = x_random(9, input_len, seed=SEED + 63)
    labels = (inputs[:, 0] + inputs[:, 1]).unsqueeze(1)
    got = engine_sr_fitness(forest, inputs.to(DEVICE), labels.to(DEVICE), True)
    assert_close_with_placement(got, torch.zeros(1), exact=True, context="SR_fitness perfect fit")


@requires_engine
def test_tree_sr_fitness_vs_reference():
    forest = make_forest(curated_edge_population(), input_len=3)
    tree = forest[3]
    inputs = x_random(N_POINTS, 3, seed=SEED + 71)
    labels = x_random(N_POINTS, 1, seed=SEED + 72)
    got = tree.SR_fitness(inputs.to(DEVICE), labels.to(DEVICE), True, "auto")
    assert tuple(got.shape) == (1,), f"Tree.SR_fitness should return (1,), got {tuple(got.shape)}"
    values, types, sizes = rows_of_tree(tree)
    want = reference.sr_fitness_from_rows(values, types, sizes, inputs, labels, use_mse=True)
    assert_close_with_placement(got, want, context="Tree.SR_fitness")


@requires_engine
def test_execute_modes_agree():
    """All documented execute_mode strings must be accepted and agree with the oracle."""
    modes = ["torch", "hybrid parallel", "data parallel", "tree parallel", "auto"]
    forest = make_forest(curated_edge_population(), input_len=3)
    inputs = x_random(N_POINTS, 3, seed=SEED + 81)
    labels = x_random(N_POINTS, 1, seed=SEED + 82)
    want = reference_sr_fitness(forest, inputs, labels, True)

    results = {}
    for mode in modes:
        got = engine_sr_fitness(forest, inputs.to(DEVICE), labels.to(DEVICE), True, mode)
        assert tuple(got.shape) == (forest.pop_size,), f"execute_mode={mode!r}: bad shape"
        assert_close_with_placement(got, want, context=f"execute_mode={mode!r}")
        results[mode] = got

    baseline = results["auto"]
    for mode, got in results.items():
        assert_close_with_placement(got, baseline, context=f"execute_mode={mode!r} vs auto")


# =========================================================================== #
# 6. Row padding / encoding corners
# =========================================================================== #
def _junk_tail(values, types, sizes, tail_value, tail_type):
    """Copy of the rows with a rewritten, unreachable tail (indices >= subtree_size[0])."""
    junk_values = values.clone()
    junk_types = types.clone()
    junk_sizes = sizes.clone()
    for i in range(values.shape[0]):
        start = int(sizes[i, 0])
        if start >= values.shape[1]:
            continue
        junk_values[i, start:] = tail_value
        junk_types[i, start:] = tail_type
        junk_sizes[i, start:] = 1
    return junk_values, junk_types, junk_sizes


@requires_engine
def test_row_tail_is_ignored():
    """Only [0, subtree_size[0]) is meaningful; the tail must not influence the result."""
    values, types, sizes = rows_from_specs(curated_edge_population())
    x = x_random(9, 3, seed=SEED + 91)

    forests = {}
    for tag, tail_value, tail_type in [
        ("nan_const", float("nan"), CONST),
        ("zeros_const", 0.0, CONST),
        ("huge_const", 1e30, CONST),
        ("junk_binary", 0.0, BFUNC),
        ("junk_ternary", 1.0, TFUNC),
    ]:
        jv, jt, js = _junk_tail(values, types, sizes, tail_value, tail_type)
        forests[tag] = Forest(3, 1, jv.to(DEVICE), jt.to(DEVICE), js.to(DEVICE))

    outputs = {tag: engine_batch_forward(f, x) for tag, f in forests.items()}
    baseline = outputs["nan_const"]
    for tag, got in outputs.items():
        assert_same_bits(got, baseline, context=f"row tail ({tag}) changes the result")
    assert_close_with_placement(baseline, reference_batch_forward(forests["nan_const"], x), context="tail baseline")


@requires_engine
def test_kernel_arity_fallback_is_preserved():
    """A function id that does not belong to the node's arity yields exactly 0.0.

    ``forward.cu::_process_node`` initializes ``float top_val{}`` and only overwrites
    it for ids handled by the current arity branch, so an invalid combination (never
    produced by generation, but defined by the kernels) evaluates to 0. A TFUNC node
    always evaluates IF regardless of its function id.
    """
    specs = [
        raw(BFUNC, float(SIN), var(0), var(1)),
        raw(BFUNC, float(IF), var(0), var(1)),
        raw(BFUNC, 30.0, var(0), var(1)),
        raw(UFUNC, float(ADD), var(0)),
        raw(UFUNC, 30.0, var(0)),
        raw(TFUNC, float(MUL), var(0), var(1), var(2)),
    ]
    forest = make_forest(specs, input_len=3)
    x = torch.tensor([[0.0, 2.0, 4.0], [-1.0, 1.0, 0.0]], dtype=torch.float32)
    got = compare_batch_forward(forest, x, context="arity fallback")
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 4.0], dtype=torch.float32)  # last: IF(0,2,4) -> 4
    # the engine runs on DEVICE (cuda when available), the hand-written expectation
    # is a CPU tensor -- compare through the shared CPU-cast helper
    torch.testing.assert_close(_to_cpu_float(got[:, 0, 0]), expected, rtol=0, atol=0)


# =========================================================================== #
# 7. Multi-output (output_len > 1) -- optional for the engine
# =========================================================================== #
_MULTI_OUT_SUPPORTED = None
_MULTI_OUT_REASON = ""


def multi_output_supported():
    global _MULTI_OUT_SUPPORTED, _MULTI_OUT_REASON
    if _MULTI_OUT_SUPPORTED is None:
        try:
            values, types, sizes = rows_from_specs([out_fn(ADD, 0, var(0), var(1))], output_len=2)
            forest = Forest(2, 2, values.to(DEVICE), types.to(DEVICE), sizes.to(DEVICE))
            out = forest.batch_forward(torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=DEVICE))
            _MULTI_OUT_SUPPORTED = tuple(out.shape) == (1, 1, 2)
            if not _MULTI_OUT_SUPPORTED:
                _MULTI_OUT_REASON = f"batch_forward returned shape {tuple(out.shape)} instead of (1, 1, 2)"
        except Exception as exc:  # the engine is allowed to refuse multi-output for now
            _MULTI_OUT_SUPPORTED = False
            _MULTI_OUT_REASON = f"{type(exc).__name__}: {exc}"
    return _MULTI_OUT_SUPPORTED


def skip_without_multi_output():
    if not multi_output_supported():
        pytest.skip(f"engine does not support output_len > 1 yet ({_MULTI_OUT_REASON})")


@requires_engine
def test_multi_output_vs_reference():
    skip_without_multi_output()
    specs = [
        fn(ADD, out_fn(ADD, 0, var(0), var(1)), out_fn(MUL, 1, var(1), var(2))),
        out_fn(ADD, 0, fn(ADD, out_fn(ADD, 0, var(0), var(1)), const(3.0)), const(0.0)),
        fn(IF, var(0), out_fn(TANH, 3, var(1)), out_fn(ABS, 1, var(2))),  # out index >= out_len -> dropped
        out_fn(ADD, 1, out_fn(MUL, 1, var(0), var(2)), const(2.0)),
        fn(ADD, var(0), const(0.0)),  # no out-node at all -> all outputs zero
    ]
    forest = make_forest(specs, input_len=3, output_len=2, pad_to=16)
    assert forest.output_len == 2
    for name, x in adversarial_inputs(input_len=3, n=16):
        compare_batch_forward(forest, x, context=f"multi-output / {name}")

    inputs = x_random(N_POINTS, 3, seed=SEED + 101)
    labels = x_random(N_POINTS, 2, seed=SEED + 102)
    got = engine_sr_fitness(forest, inputs.to(DEVICE), labels.to(DEVICE), True)
    assert_close_with_placement(
        got, reference_sr_fitness(forest, inputs, labels, True), context="multi-output SR_fitness"
    )


@requires_engine
def test_multi_output_generated_population_vs_reference():
    skip_without_multi_output()
    descriptor, forest = make_random_population(
        max(TREES // 5, 50),
        SEED,
        max_tree_len=64,
        input_len=3,
        output_len=3,
        using_funcs=["+", "-", "*", "/", "sin", "cos", "max", "min"],
        out_prob=0.6,
    )
    assert forest.output_len == 3
    x = x_random(N_POINTS, 3, seed=SEED + 111)
    compare_batch_forward(forest, x, context="generated multi-output population")


# =========================================================================== #
# 8. Golden fixtures
#
#    (a) "engine" schema       -- this package's own fixtures (tests/goldens/*.npz),
#                                 captured from the ETL engine; see the module
#                                 docstring for the array names.
#    (b) "legacy-capture" schema -- <repo-root>/test/goldens/*.npz, the frozen
#                                 outputs of the deleted custom CUDA kernels
#                                 (test/golden_capture.py, documented in
#                                 test/goldens/README.md).  Rebuilding the Forest
#                                 from the stored arrays and comparing the engine
#                                 against the stored outputs cross-validates the
#                                 rewrite against the kernels it replaced.
#
#    Files matching neither schema (foreign fixtures dropped into a golden dir)
#    are reported through a warning with an explicit reason and never fail the
#    suite; the same holds for individual arrays a fixture does not store.
# =========================================================================== #
GOLDEN_SCHEMA_ENGINE = "engine"
GOLDEN_SCHEMA_LEGACY_CAPTURE = "legacy-capture"

_ENGINE_SCHEMA_TREE_KEYS = ("node_value", "node_type", "subtree_size")

#: Legacy captures store one `(prefix)__node_value/_node_type/_subtree_size` triple per
#: stage: `forest__*` (evaluation stages), `initial_forest__*`, `final_forest__*` and
#: `best_tree__*` (training stage).
_LEGACY_TREE_SUFFIXES = ("__node_value", "__node_type", "__subtree_size")

#: test/goldens/README.md "Recommended comparison tolerances": node_type /
#: subtree_size / headers exactly, floats with rtol = atol = 1e-4.  The captured
#: kernels were built with `-use_fast_math` and the four execute modes reduce in
#: different orders, so bit-equality is explicitly NOT the contract.
LEGACY_RTOL = 1e-4
LEGACY_ATOL = 1e-4

#: The four kernel execute modes; the capture spells them with `_` in array names.
_LEGACY_EXECUTE_MODES = ("hybrid parallel", "data parallel", "tree parallel", "auto")

#: Frozen capture of the pre-ETL engine (test/goldens/).
_LEGACY_CAPTURE_DIR = _REPO_ROOT / "test" / "goldens"


def golden_directories():
    """Golden directories scanned for fixtures, in priority order and deduplicated."""
    directories = []
    env_dir = os.environ.get("EVOGP_GOLDEN_DIR")
    if env_dir:
        directories.append(pathlib.Path(env_dir))
    directories.append(_LEGACY_CAPTURE_DIR)
    directories.append(_HERE / "goldens")

    seen, unique = set(), []
    for directory in directories:
        key = str(directory.resolve()) if directory.is_dir() else str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def golden_paths():
    paths = []
    for directory in golden_directories():
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.npz")))
    return paths


def _legacy_tree_prefixes(keys):
    """Tree-tensor prefixes (``forest``, ``initial_forest``, ...) present in a fixture."""
    prefixes = []
    for key in keys:
        if not key.endswith("__node_value"):
            continue
        prefix = key[: -len("__node_value")]
        if all(f"{prefix}{suffix}" in keys for suffix in _LEGACY_TREE_SUFFIXES):
            prefixes.append(prefix)
    return sorted(prefixes)


def classify_golden_fixture(data):
    """``(schema, reason)`` of an opened ``.npz``; ``schema`` is ``None`` when unknown.

    The reason is always a non-empty explanation (which keys were found / which are
    missing) so the caller can report *why* a fixture could not be checked.
    """
    keys = set(data.files)
    if all(key in keys for key in _ENGINE_SCHEMA_TREE_KEYS):
        found = [key for key in (*_ENGINE_SCHEMA_TREE_KEYS, "x", "labels", "expected",
                                 "expected_fitness", "use_mse") if key in keys]
        return GOLDEN_SCHEMA_ENGINE, "engine schema, keys: " + ", ".join(found)
    prefixes = _legacy_tree_prefixes(keys)
    if prefixes:
        return GOLDEN_SCHEMA_LEGACY_CAPTURE, "legacy kernel-capture schema, populations: " + ", ".join(
            f"{prefix}__*" for prefix in prefixes
        )
    listing = sorted(keys)
    if len(listing) > 12:
        listing = listing[:12] + ["..."]
    return None, (
        "matches neither golden schema (engine keys missing "
        f"{[key for key in _ENGINE_SCHEMA_TREE_KEYS if key not in keys]}, no "
        f"'<prefix>{_LEGACY_TREE_SUFFIXES[0]}/{'/'.join(_LEGACY_TREE_SUFFIXES[1:])}' triple either); "
        f"npz keys = {listing}"
    )


def collect_golden_fixtures():
    """``[(path, schema_or_None, reason)]`` for every ``*.npz`` on the scan path.

    A file that cannot even be opened is reported as unrecognized (with the
    exception text) instead of raising: the golden directories may legitimately
    contain fixtures produced by other tooling.
    """
    discovered = []
    for path in golden_paths():
        try:
            with np.load(path) as data:
                schema, reason = classify_golden_fixture(data)
        except Exception as exc:  # noqa: BLE001 - foreign/corrupt file: report, never fail
            schema, reason = None, f"could not be read as an .npz ({exc!r})"
        discovered.append((path, schema, reason))
    return discovered


def report_unrecognized_fixtures(discovered):
    """Warn -- never fail -- about every discovered fixture that will not be checked."""
    for path, schema, reason in discovered:
        if schema is None:
            warnings.warn(f"golden fixture {path.name!r} was NOT checked: {reason}", stacklevel=2)


class _GoldenOutcome:
    """Bookkeeping of one golden test: which comparisons ran, which did not (and why).

    ``finish()`` warns about every skipped comparison and turns a test that ran no
    comparison at all into an explicit ``pytest.skip``, so a fixture can never be
    ignored silently.
    """

    def __init__(self, context):
        self.context = context
        self.checked = []
        self.unchecked = []

    def done(self, what):
        self.checked.append(what)

    def skip(self, what, why):
        self.unchecked.append(f"{what} ({why})")

    def finish(self):
        if self.unchecked:
            warnings.warn(
                f"{self.context}: {len(self.unchecked)} comparison(s) not performed -- "
                + "; ".join(self.unchecked),
                stacklevel=2,
            )
        if not self.checked:
            pytest.skip(f"{self.context}: nothing to compare ({'; '.join(self.unchecked) or 'no fixtures'})")
        return len(self.checked)


def golden_array(data, key, outcome, what=None):
    """One stored array, or ``None`` (recorded as not compared, with an explicit reason)."""
    if key in data.files:
        return np.asarray(data[key])
    outcome.skip(what or key, f"{key} is not stored in this fixture")
    return None


def golden_tensor(array, dtype):
    return torch.as_tensor(np.asarray(array), dtype=dtype).to(DEVICE)


def load_legacy_capture(filename):
    """Open one frozen ``test/goldens/*.npz`` capture, or skip with a reason.

    The capture is an *optional* cross-check against the deleted kernels: an absent
    file, or one that does not carry the documented schema, is a skip (never a
    failure) and says so explicitly.
    """
    path = _LEGACY_CAPTURE_DIR / filename
    if not path.is_file():
        pytest.skip(f"legacy kernel capture {path} is not available (optional cross-check)")
    data = np.load(path)
    schema, reason = classify_golden_fixture(data)
    if schema != GOLDEN_SCHEMA_LEGACY_CAPTURE:
        data.close()
        pytest.skip(f"legacy kernel capture {path.name} was not recognized: {reason}")
    return path, data


def legacy_forest(data, input_len, output_len, prefix="forest"):
    """The frozen population of a legacy capture, as a ``Forest`` on ``DEVICE``.

    The capture zero-filled the dead tail of every row (README "Dead region of the
    tree tensors is normalised"); the engine only reads ``[0, subtree_size[0])``,
    so the tail is a documented don't-care.
    """
    return Forest(
        input_len,
        output_len,
        golden_tensor(data[f"{prefix}__node_value"], torch.float32),
        golden_tensor(data[f"{prefix}__node_type"], torch.int16),
        golden_tensor(data[f"{prefix}__subtree_size"], torch.int16),
    )


@requires_engine
def test_golden_fixture_discovery():
    """Fixtures are discovered and classified by SCHEMA; nothing here can fail the suite.

    ``test/goldens/`` holds the frozen pre-ETL kernel capture and
    ``src/evogp/tests/goldens/`` this package's engine fixtures, but the two are told
    apart by their array names, never by their location: a fixture copied between
    directories (the two READMEs invite exactly that) is still handled -- or
    reported -- and never fed to a loader that would mis-read it.  Every file that
    is *not* going to be compared is listed in the warnings summary with its reason.
    """
    discovered = collect_golden_fixtures()
    if not discovered:
        pytest.skip(
            "no golden fixtures found (searched $EVOGP_GOLDEN_DIR, test/goldens/, src/evogp/tests/goldens/)"
        )
    report_unrecognized_fixtures(discovered)

    recognized = [(path, schema) for path, schema, _ in discovered if schema is not None]
    if not recognized:
        pytest.skip(
            "golden fixtures found but none matches a known schema: "
            + "; ".join(f"{path.name} ({reason})" for path, _, reason in discovered)
        )

    # Directory convention, informational only (see the docstring): the check is
    # about the fixture set, and a copy in the "wrong" directory is still checked
    # correctly or reported by this test.
    conventions = {
        str(_LEGACY_CAPTURE_DIR.resolve()): GOLDEN_SCHEMA_LEGACY_CAPTURE,
        str((_HERE / "goldens").resolve()): GOLDEN_SCHEMA_ENGINE,
    }
    for path, schema in recognized:
        wanted = conventions.get(str(path.parent.resolve()))
        if wanted is not None and schema != wanted:
            warnings.warn(
                f"golden fixture {path.name!r} carries the {schema!r} schema but lives in the directory "
                f"usually reserved for the {wanted!r} schema; it is still handled by schema",
                stacklevel=2,
            )


@requires_engine
def test_golden_engine_schema_fixtures():
    """Engine schema (this package's fixtures): batch_forward / SR_fitness expectations."""
    paths = [path for path, schema, _ in collect_golden_fixtures() if schema == GOLDEN_SCHEMA_ENGINE]
    if not paths:
        pytest.skip("no engine-schema golden fixtures found (looked in $EVOGP_GOLDEN_DIR, tests/goldens/)")
    outcome = _GoldenOutcome("engine-schema golden fixtures")
    for path in paths:
        with np.load(path) as data:
            x = golden_array(data, "x", outcome, f"{path.name}/x")
            if x is None:
                continue
            x = golden_tensor(x, torch.float32)
            values = golden_tensor(data["node_value"], torch.float32)
            types = golden_tensor(data["node_type"], torch.int16)
            sizes = golden_tensor(data["subtree_size"], torch.int16)
            use_mse = bool(np.asarray(data["use_mse"]).reshape(-1)[0]) if "use_mse" in data else True

            if "expected" in data:
                expected = golden_tensor(data["expected"], torch.float32)
                forest = Forest(int(x.shape[1]), int(expected.shape[-1]), values, types, sizes)
                assert_close_with_placement(
                    forest.batch_forward(x), expected, context=f"golden {path.name}/expected"
                )
                outcome.done(f"{path.name}/expected")
            if "expected_fitness" in data:
                labels = golden_array(data, "labels", outcome, f"{path.name}/labels")
                if labels is None:
                    continue
                labels = golden_tensor(labels, torch.float32)
                expected_fitness = golden_tensor(data["expected_fitness"], torch.float32)
                forest = Forest(int(x.shape[1]), int(labels.shape[1]), values, types, sizes)
                got = engine_sr_fitness(forest, x, labels, use_mse)
                assert_close_with_placement(
                    got, expected_fitness, context=f"golden {path.name}/expected_fitness"
                )
                outcome.done(f"{path.name}/expected_fitness")
    outcome.finish()


@requires_engine
def test_golden_legacy_capture_forward():
    """tree/forest/forest-batch forward on the frozen pre-ETL population.

    ``test/goldens/forward_eval.npz`` stores the population the kernels evaluated;
    rebuilding it involves no RNG, so this compares the new engine against the old
    kernels on identical inputs (README "How to use these goldens" step 2).  The
    display strings are captured alongside and pin the same trees' rendering.
    """
    path, data = load_legacy_capture("forward_eval.npz")
    outcome = _GoldenOutcome(f"legacy capture {path.name}")
    try:
        x = golden_tensor(data["x"], torch.float32)
        labels = golden_tensor(data["labels"], torch.float32)
        forest = legacy_forest(data, int(x.shape[1]), int(labels.shape[1]))

        expected = golden_array(data, "forest_forward", outcome)
        if expected is not None:
            assert_close_with_placement(
                forest.forward(x),
                golden_tensor(expected, torch.float32),
                rtol=LEGACY_RTOL,
                atol=LEGACY_ATOL,
                context=f"{path.name}/forest_forward",
            )
            outcome.done("forest_forward")

        x_small = golden_array(data, "x_small", outcome)
        expected = golden_array(data, "forest_batch_forward", outcome)
        if x_small is not None and expected is not None:
            assert_close_with_placement(
                forest.batch_forward(golden_tensor(x_small, torch.float32)),
                golden_tensor(expected, torch.float32),
                rtol=LEGACY_RTOL,
                atol=LEGACY_ATOL,
                context=f"{path.name}/forest_batch_forward",
            )
            outcome.done("forest_batch_forward")

        expected = golden_array(data, "tree_forward", outcome)
        if expected is not None:
            trees = [forest[i].forward(x) for i in range(int(expected.shape[0]))]
            assert_close_with_placement(
                torch.stack(trees, dim=0),
                golden_tensor(expected, torch.float32),
                rtol=LEGACY_RTOL,
                atol=LEGACY_ATOL,
                context=f"{path.name}/tree_forward",
            )
            outcome.done("tree_forward")

        # Display-only strings ("informational" in the README's table), checked
        # exactly: they are pure functions of the stored rows, so a difference is a
        # rendering change, not a numeric one.
        from evogp.tree.utils import str_tree

        renderers = {
            "tree_infix": lambda tree: tree.to_infix(),
            "tree_sympy_str": lambda tree: str(tree.to_sympy_expr()),
            "tree_preorder_str": lambda tree: str_tree(
                tree.node_value.detach().cpu().numpy(),
                tree.node_type.detach().cpu().numpy(),
                tree.subtree_size.detach().cpu().numpy(),
            ),
        }
        for key, render in renderers.items():
            expected = golden_array(data, key, outcome)
            if expected is None:
                continue
            got = np.array([render(forest[i]) for i in range(len(expected))])
            assert np.array_equal(got, expected), (
                f"{path.name}/{key}: display strings differ\n  engine: {list(got)}\n  golden: {list(expected)}"
            )
            outcome.done(key)
    finally:
        data.close()
    outcome.finish()


@requires_engine
def test_golden_legacy_capture_generation():
    """``forest_random.npz``: the generation-stage capture, used as a frozen input.

    The population is deliberately NOT regenerated: the rewrite samples with the
    torch RNG (distribution-equivalent to the legacy stream, not bit-identical), so
    ``test/goldens/README.md`` step 1 allows a shape/legality use only.  What is
    checkable without any RNG is that this file carries the very population the
    evaluation stages ran on (documented "bitwise identical arrays, also stored here
    for self-containment") -- and that the engine reproduces ``forward_eval.npz``'s
    stored ``forest_batch_forward`` from *these* rows, which is the one expectation
    the generation stage can be measured against.
    """
    path, data = load_legacy_capture("forest_random.npz")
    outcome = _GoldenOutcome(f"legacy capture {path.name}")
    reference_path = _LEGACY_CAPTURE_DIR / "forward_eval.npz"
    try:
        if not reference_path.is_file():
            pytest.skip(f"{reference_path} is not available (optional cross-check)")
        for key in ("forest__node_value", "forest__node_type", "forest__subtree_size",
                    "x", "labels", "x_small"):
            mine = golden_array(data, key, outcome, f"{path.name}/{key}")
            if mine is None:
                continue
            with np.load(reference_path) as other:
                theirs = golden_array(other, key, outcome, f"{reference_path.name}/{key}")
            if theirs is None:
                continue
            assert np.array_equal(mine, theirs), (
                f"{path.name}/{key} differs from {reference_path.name}/{key}, but both stages document "
                "bitwise-identical inputs; the fixture set is inconsistent"
            )
            outcome.done(f"{key} == {reference_path.name}")

        x = golden_tensor(data["x"], torch.float32)
        labels = golden_tensor(data["labels"], torch.float32)
        forest = legacy_forest(data, int(x.shape[1]), int(labels.shape[1]))

        # The capture's own data definition (README): x[i] = ((i % 8 - 3.5)/2, (i // 8 - 3.5)/2)
        # and labels = x0**2 - x1; x_small is the first 8 rows.
        assert_close_with_placement(
            labels,
            (x[:, 0:1] ** 2 - x[:, 1:2]),
            rtol=LEGACY_RTOL,
            atol=LEGACY_ATOL,
            context=f"{path.name}/labels == x0**2 - x1",
        )
        outcome.done("labels == x0**2 - x1")
        assert np.array_equal(np.asarray(data["x_small"]), np.asarray(data["x"])[: int(data["x_small"].shape[0])]), (
            f"{path.name}: x_small is not the first rows of x"
        )
        outcome.done("x_small == x[:8]")

        with np.load(reference_path) as other:
            expected = golden_array(other, "forest_batch_forward", outcome)
            if expected is not None:
                x_small = golden_tensor(data["x_small"], torch.float32)
                assert_close_with_placement(
                    forest.batch_forward(x_small),
                    golden_tensor(expected, torch.float32),
                    rtol=LEGACY_RTOL,
                    atol=LEGACY_ATOL,
                    context=f"{path.name} rows -> {reference_path.name}/forest_batch_forward",
                )
                outcome.done("batch_forward on the generation capture")

        outcome.skip(
            "forest__node_* regeneration from descriptor__*/seed",
            "the rewrite samples with the torch RNG (distribution-equivalent, not bit-identical); "
            "test/goldens/README.md step 1",
        )
    finally:
        data.close()
    outcome.finish()


@requires_engine
@requires_problem
def test_golden_legacy_capture_sr_fitness():
    """SR_fitness in every kernel execute mode (MSE + MAE) and the problem-level path.

    ``test/goldens/sr_fitness.npz`` stores, per mode, ``forest.SR_fitness`` (64,),
    the first 8 ``Tree.SR_fitness`` values and ``SymbolicRegression.evaluate`` (64,),
    plus the mode-independent pure-torch problem path.  README semantics recovered
    from those arrays: ``SR_fitness`` returns the POSITIVE loss (MSE / MAE of the
    forward pass) and ``SymbolicRegression.evaluate`` is exactly its negation.  The
    array names spell the modes with ``_``.
    """
    path, data = load_legacy_capture("sr_fitness.npz")
    outcome = _GoldenOutcome(f"legacy capture {path.name}")
    try:
        x = golden_tensor(data["x"], torch.float32)
        labels = golden_tensor(data["labels"], torch.float32)
        forest = legacy_forest(data, int(x.shape[1]), int(labels.shape[1]))

        for mode in _LEGACY_EXECUTE_MODES:
            key = mode.replace(" ", "_")
            for use_mse, suffix in ((True, "mse"), (False, "nonmse")):
                what = f"forest_sr_fitness__{key}__{suffix}"
                expected = golden_array(data, what, outcome)
                if expected is not None:
                    assert_close_with_placement(
                        engine_sr_fitness(forest, x, labels, use_mse, mode),
                        golden_tensor(expected, torch.float32),
                        rtol=LEGACY_RTOL,
                        atol=LEGACY_ATOL,
                        context=f"{path.name}/{what}",
                    )
                    outcome.done(what)

                what = f"tree_sr_fitness__{key}__{suffix}"
                expected = golden_array(data, what, outcome)
                if expected is not None:
                    got = torch.cat(
                        [
                            forest[i].SR_fitness(x, labels, use_mse, mode)
                            for i in range(int(expected.shape[0]))
                        ]
                    )
                    assert_close_with_placement(
                        got,
                        golden_tensor(expected, torch.float32),
                        rtol=LEGACY_RTOL,
                        atol=LEGACY_ATOL,
                        context=f"{path.name}/{what}",
                    )
                    outcome.done(what)

                # `SymbolicRegression.evaluate` is the negated loss of the same call.
                for metric, mse in (("mse", True), ("nonmse", False)):
                    what = f"problem_evaluate__{key}__{metric}"
                    expected = golden_array(data, what, outcome)
                    if expected is None:
                        continue
                    problem = SymbolicRegression(
                        datapoints=x.clone(), labels=labels.clone(), execute_mode=mode
                    )
                    got = problem.evaluate(forest, use_MSE=mse)
                    assert_close_with_placement(
                        got,
                        golden_tensor(expected, torch.float32),
                        rtol=LEGACY_RTOL,
                        atol=LEGACY_ATOL,
                        context=f"{path.name}/{what}",
                    )
                    outcome.done(what)

        what = "problem_evaluate__torch__mse"
        expected = golden_array(data, what, outcome)
        if expected is not None:
            problem = SymbolicRegression(
                datapoints=x.clone(), labels=labels.clone(), execute_mode="torch"
            )
            assert_close_with_placement(
                problem.evaluate(forest),
                golden_tensor(expected, torch.float32),
                rtol=LEGACY_RTOL,
                atol=LEGACY_ATOL,
                context=f"{path.name}/{what}",
            )
            outcome.done(what)
    finally:
        data.close()
    outcome.finish()


@requires_engine
@requires_problem
def test_golden_legacy_capture_training():
    """The RNG-independent parts of the 12-generation reference (``training.npz``).

    README "How to use these goldens" step 4:
    * ``initial_fitness`` must be reproducible from the stored ``initial_forest__*``
      + ``datapoints``/``labels`` alone -- checked here through
      ``SymbolicRegression.evaluate``.  ``generation_fitness`` records what
      ``StandardPipeline.step`` returned, i.e. *after* its ``NaN -> -inf``
      replacement, so the same convention is applied before comparing.
    * the generation-by-generation trajectory (``generation_fitness[1:]``,
      ``final_forest__*``) additionally needs the legacy kernels' exact
      RNG-consumption order and is therefore reported, not reproduced;
    * ``best_tree`` is a stored population row as well, so its fitness must equal
      the stored final ``best_fitness_per_generation`` -- a second, independent
      (and much smaller) confirmation of the whole trajectory's endpoint.
    """
    path, data = load_legacy_capture("training.npz")
    outcome = _GoldenOutcome(f"legacy capture {path.name}")
    try:
        # Structural invariants of the frozen trajectory (exact, per the README's
        # tolerance table) -- these must hold before any numbers are compared.
        generation_fitness = np.asarray(data["generation_fitness"])
        initial_fitness = np.asarray(data["initial_fitness"])
        best_fitness = np.asarray(data["best_fitness_per_generation"])
        assert np.array_equal(initial_fitness, generation_fitness[0]), "initial_fitness != generation_fitness[0]"
        assert np.all(np.diff(best_fitness.astype(np.float64)) >= 0.0), "best_fitness_per_generation is not monotone"
        assert not np.isnan(generation_fitness).any(), "the pipeline replaces NaN fitness with -inf; none may remain"

        datapoints = golden_tensor(data["datapoints"], torch.float32)
        labels = golden_tensor(data["labels"], torch.float32)
        problem = SymbolicRegression(
            datapoints=datapoints.clone(), labels=labels.clone(), execute_mode="auto"
        )

        what = "initial_fitness from initial_forest"
        forest = legacy_forest(
            data, int(datapoints.shape[1]), int(labels.shape[1]), prefix="initial_forest"
        )
        got = problem.evaluate(forest)
        got[torch.isnan(got)] = -torch.inf  # StandardPipeline.step's convention
        assert_close_with_placement(
            got,
            golden_tensor(initial_fitness, torch.float32),
            rtol=LEGACY_RTOL,
            atol=LEGACY_ATOL,
            context=f"{path.name}/{what}",
        )
        outcome.done(what)

        what = "best_tree fitness vs best_fitness_per_generation[-1]"
        forest = legacy_forest(data, int(datapoints.shape[1]), int(labels.shape[1]), prefix="best_tree")
        got = problem.evaluate(forest)
        assert_close_with_placement(
            got,
            golden_tensor(best_fitness[-1:], torch.float32),
            rtol=LEGACY_RTOL,
            atol=LEGACY_ATOL,
            context=f"{path.name}/{what}",
        )
        outcome.done(what)

        outcome.skip(
            "generation_fitness[1:] / final_forest__*",
            "the legacy RNG-consumption order (torch.randint keys, mutation masks, "
            "const_samples draws) is not part of the rewrite; README step 4",
        )
    finally:
        data.close()
    outcome.finish()
