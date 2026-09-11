"""Public-API compatibility tests for EvoGP after the engine rewrite.

The CUDA extension is being replaced by an ETL-based evaluation engine plus plain
torch generation / mutation / crossover.  Everything in this file is asserted
against the *public* surface that the rest of the library, the examples and the
tutorials import, so that the rewrite stays drop-in:

* the import surface of ``evogp.tree`` and ``evogp.problem`` and the empty-
  ``__init__`` semantics of ``import evogp``,
* ``Forest`` container dunders (indexing views, slicing, iteration, concat,
  pickle, repr) and ``zero_generate``,
* ``Tree`` symbolic export (``to_sympy_expr`` / ``to_infix`` / ``to_png``)
  cross-checked against ``forward`` (constant-only trees included),
* the documented ``execute_mode`` strings of ``forward`` / ``SR_fitness``
  (results must agree across modes; unknown strings must raise),
* module helpers (``randint``, ``MAX_STACK``, ``NType``) and the
  ``GenerateDescriptor`` construction surface.

Run from the repository root::

    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q
"""

from __future__ import annotations

import importlib.util
import inspect
import pickle
import shutil
import subprocess
import sys

import pytest
import sympy as sp
import torch

try:  # importable only once the ETL-based engine has replaced the CUDA extension
    import evogp
    from evogp.tree import (
        CombinedForest,
        CombinedTree,
        Forest,
        GenerateDescriptor,
        MAX_STACK,
        NType,
        Tree,
        randint,
    )

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only in pre-merge worktrees
    evogp = Forest = Tree = GenerateDescriptor = CombinedForest = CombinedTree = None
    MAX_STACK = NType = randint = None
    _IMPORT_ERROR = exc


@pytest.fixture(autouse=True)
def _require_engine():
    require_engine()


def require_engine():
    """Fail fast (with one clear message) while the engine is not importable."""
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "evogp.tree is not importable (the ETL-based engine is not present in this "
            f"worktree yet): {_IMPORT_ERROR!r}"
        )


NUM_FUNCS = 29  # Func.END: the length of the descriptor's roulette vector
REQUIRED_SR_MODES = ["hybrid parallel", "data parallel", "tree parallel", "auto"]
OPTIONAL_SR_MODES = ["torch"]
FORWARD_MODES = ["torch", "hybrid parallel", "data parallel", "tree parallel", "auto"]


# --------------------------------------------------------------------------- #
# Fixtures: a small, fast, NaN-free population (only +, -, * and variables)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def polynomial_descriptor():
    require_engine()
    return GenerateDescriptor(
        max_tree_len=16,
        input_len=2,
        output_len=1,
        const_prob=0.0,
        out_prob=0.5,
        using_funcs=["+", "-", "*"],
        max_layer_cnt=3,
        layer_leaf_prob=0.3,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
    )


@pytest.fixture(scope="module")
def small_population(polynomial_descriptor):
    torch.manual_seed(0)
    return Forest.random_generate(pop_size=3, descriptor=polynomial_descriptor)


@pytest.fixture(scope="module")
def small_tree(small_population):
    return small_population[0]


def api_device() -> torch.device:
    return Forest.zero_generate(1, 8, 1, 1).batch_node_value.device


def tensor_device(obj) -> torch.device:
    """Device of a ``Forest`` (batched tensors) or a ``Tree`` (plain tensors)."""
    tensor = getattr(obj, "batch_node_value", None)
    if tensor is None:
        tensor = obj.node_value
    return tensor.device


def sr_data(container, num_data: int = 6):
    device = tensor_device(container)
    inputs = torch.rand(num_data, container.input_len, device=device)
    labels = torch.rand(num_data, container.output_len, device=device)
    return inputs, labels


def uses_execute_mode(func) -> bool:
    """Whether ``func`` exposes an ``execute_mode`` keyword (or ``**kwargs``)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins without an introspectable signature
        return False
    if "execute_mode" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# --------------------------------------------------------------------------- #
# Import surface
# --------------------------------------------------------------------------- #
def test_tree_layer_import_surface():
    for cls in (GenerateDescriptor, Tree, Forest, CombinedForest, CombinedTree):
        assert isinstance(cls, type), f"{cls!r} is not a class"
    assert callable(randint)
    assert isinstance(MAX_STACK, int) and MAX_STACK >= 1024
    assert (NType.VAR, NType.CONST, NType.UFUNC, NType.BFUNC, NType.TFUNC) == (0, 1, 2, 3, 4)
    assert (NType.TYPE_MASK, NType.OUT_NODE) == (0x7F, 1 << 7)
    # max_tree_len up to MAX_STACK must still be accepted by the descriptor
    descriptor = GenerateDescriptor(
        MAX_STACK, 2, 1, using_funcs=["+"], max_layer_cnt=3, const_range=(-1.0, 1.0), sample_cnt=4
    )
    assert descriptor.max_tree_len == MAX_STACK


def test_import_evogp_is_side_effect_free():
    """``import evogp`` must succeed and must not pull in any submodule."""
    code = (
        "import sys, evogp;"
        "print(','.join(sorted(m for m in sys.modules if m.startswith('evogp'))))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"`import evogp` failed:\n{proc.stderr}"
    loaded = proc.stdout.strip()
    assert loaded == "evogp", f"`import evogp` must not import submodules, got {loaded!r}"


def test_problem_layer_import_surface():
    """``evogp.problem`` must re-export every documented problem class.

    The import is done inside the test (it is not part of this module's top-level
    surface) and must work without jax/brax/genesis: the robotics problems import
    those heavy dependencies lazily, inside their constructors only.
    """
    problem_module = importlib.import_module("evogp.problem")
    for name in (
        "BaseProblem",
        "SymbolicRegression",
        "Classification",
        "Transformation",
        "BraxProblem",
        "GenesisProblem",
        "MujocoProblem",
        "CustomLoss",
    ):
        cls = getattr(problem_module, name, None)
        assert isinstance(cls, type), f"evogp.problem.{name} is not a class: {cls!r}"

    # the re-export must be the same object as the defining submodule's attribute
    custom_loss_module = importlib.import_module("evogp.problem.custom_loss")
    assert problem_module.CustomLoss is custom_loss_module.CustomLoss


# --------------------------------------------------------------------------- #
# Forest container dunders
# --------------------------------------------------------------------------- #
def test_forest_len_and_shapes(small_population, polynomial_descriptor):
    forest = small_population
    assert len(forest) == forest.pop_size == 3
    assert forest.max_tree_len == polynomial_descriptor.max_tree_len
    assert (forest.input_len, forest.output_len) == (2, 1)
    assert forest.batch_node_value.shape == (3, 16)
    assert forest.batch_node_type.shape == (3, 16)
    assert forest.batch_subtree_size.shape == (3, 16)


def test_forest_getitem_returns_tree_view(small_population):
    for i in range(len(small_population)):
        tree = small_population[i]
        assert isinstance(tree, Tree)
        # basic indexing must keep sharing storage with the population (views, not copies)
        assert tree.node_value.data_ptr() == small_population.batch_node_value[i].data_ptr()
        assert tree.node_type.data_ptr() == small_population.batch_node_type[i].data_ptr()
        assert tree.subtree_size.data_ptr() == small_population.batch_subtree_size[i].data_ptr()
        assert tree.max_tree_len == small_population.max_tree_len


def test_forest_slice_returns_forest_view(small_population):
    sliced = small_population[1:3]
    assert isinstance(sliced, Forest)
    assert sliced.pop_size == 2
    assert sliced.max_tree_len == small_population.max_tree_len
    assert sliced.batch_node_value.data_ptr() == small_population.batch_node_value[1:3].data_ptr()
    for i, j in enumerate(range(1, 3)):
        assert torch.equal(
            sliced.batch_node_value[i].cpu(), small_population.batch_node_value[j].cpu()
        )


def test_forest_iteration_matches_indexing(small_population):
    trees = list(small_population)
    assert len(trees) == len(small_population)
    for i, tree in enumerate(trees):
        assert isinstance(tree, Tree)
        assert torch.equal(tree.node_value.cpu(), small_population[i].node_value.cpu())
        assert torch.equal(tree.node_type.cpu(), small_population[i].node_type.cpu())


def test_forest_concat_with_forest_and_tree(small_population):
    doubled = small_population + small_population
    assert isinstance(doubled, Forest)
    assert doubled.pop_size == 2 * small_population.pop_size
    assert torch.equal(
        doubled.batch_node_value.cpu(), small_population.batch_node_value.repeat(2, 1).cpu()
    )

    tree = small_population[0]
    extended = small_population + tree
    assert isinstance(extended, Forest)
    assert extended.pop_size == small_population.pop_size + 1
    assert torch.equal(
        extended.batch_node_value[-1].cpu(), small_population.batch_node_value[0].cpu()
    )


def test_forest_pickle_roundtrip(small_population):
    restored = pickle.loads(pickle.dumps(small_population))

    assert isinstance(restored, Forest)
    assert (restored.pop_size, restored.max_tree_len) == (
        small_population.pop_size,
        small_population.max_tree_len,
    )
    assert (restored.input_len, restored.output_len) == (2, 1)
    for i in range(restored.pop_size):
        total = int(small_population.batch_subtree_size[i, 0])
        assert torch.equal(
            restored.batch_node_value[i, :total].cpu(),
            small_population.batch_node_value[i, :total].cpu(),
        )
        assert torch.equal(
            restored.batch_node_type[i, :total].cpu(),
            small_population.batch_node_type[i, :total].cpu(),
        )
        assert torch.equal(
            restored.batch_subtree_size[i, :total].cpu(),
            small_population.batch_subtree_size[i, :total].cpu(),
        )
    # the restored population must still be evaluable
    x = torch.rand(
        restored.pop_size, restored.input_len, device=restored.batch_node_value.device
    )
    assert torch.allclose(
        restored.forward(x).detach().cpu().float(),
        small_population.forward(
            x.to(small_population.batch_node_value.device)
        ).detach().cpu().float(),
        rtol=1e-4,
        atol=1e-5,
    )


def test_forest_repr_smoke(small_population):
    """``repr`` builds a sympy expression per tree - keep it to a tiny population."""
    tiny = small_population[0:1]
    text = repr(tiny)
    assert isinstance(text, str) and text
    assert isinstance(str(tiny), str)


def test_forest_zero_generate_encoding():
    forest = Forest.zero_generate(4, 8, 2, 1)
    assert (forest.pop_size, forest.max_tree_len) == (4, 8)
    assert forest.input_len == 2 and forest.output_len == 1
    assert torch.equal(
        forest.batch_subtree_size[:, 0],
        torch.ones(4, dtype=torch.int16).to(forest.batch_subtree_size.device),
    )
    assert torch.equal(
        forest.batch_node_type[:, 0],
        torch.full((4,), NType.CONST, dtype=torch.int16).to(forest.batch_node_type.device),
    )
    assert torch.equal(
        forest.batch_node_value[:, 0],
        torch.zeros(4, dtype=torch.float32).to(forest.batch_node_value.device),
    )
    x = torch.rand(4, 2, device=forest.batch_node_value.device)
    output = forest.forward(x)
    assert tuple(output.shape) == (4, 1)
    assert torch.allclose(output, torch.zeros_like(output))


# --------------------------------------------------------------------------- #
# Tree: symbolic export vs. forward
# --------------------------------------------------------------------------- #
def test_tree_to_sympy_expr_matches_forward(small_tree):
    expr = small_tree.to_sympy_expr()
    assert isinstance(expr, sp.Expr)
    assert expr.free_symbols <= set(sp.symbols(f"x:{small_tree.input_len}", real=True))

    symbols = sp.symbols(f"x:{small_tree.input_len}", real=True)
    device = small_tree.node_value.device
    points = torch.rand(4, small_tree.input_len, dtype=torch.float32, device=device)
    output = small_tree.forward(points)
    assert tuple(output.shape) == (4, small_tree.output_len)

    for row in range(points.shape[0]):
        subs = {symbols[i]: float(points[row, i].item()) for i in range(small_tree.input_len)}
        expected = float(sp.N(expr.subs(subs)))
        got = float(output[row, 0].item())
        assert got == pytest.approx(expected, rel=1e-3, abs=1e-4)


def test_tree_forward_one_dimensional_input(small_tree):
    point = torch.rand(small_tree.input_len, dtype=torch.float32, device=small_tree.node_value.device)
    result = small_tree.forward(point)
    assert result.numel() == small_tree.output_len
    batched = small_tree.forward(point.unsqueeze(0))
    assert float(result.reshape(-1)[0]) == pytest.approx(float(batched[0, 0]), rel=1e-6, abs=1e-6)


def test_tree_to_infix_smoke(small_tree):
    infix = small_tree.to_infix()
    assert isinstance(infix, str) and infix
    assert "x" in infix


def mk_const(value, input_len=1, max_len=16, device="cuda"):
    """A single-node (CONST-only) tree: ``subtree_size[0] == 1`` marks the leaf."""
    nv = torch.zeros(max_len, dtype=torch.float32, device=device)
    nt = torch.zeros(max_len, dtype=torch.int16, device=device)
    st = torch.zeros(max_len, dtype=torch.int16, device=device)
    nv[0], nt[0], st[0] = value, NType.CONST, 1
    return Tree(input_len, 1, nv, nt, st)


def test_tree_constant_only_repr_and_sympy():
    """A constant-only tree must render as a plain sympy number.

    Regression test: ``to_sympy_expr`` used to push the raw numpy scalar onto the
    expression stack, so ``repr`` printed ``np.float32(0.25018632)`` and
    ``to_sympy_expr`` returned a ``numpy.float32`` instead of a sympy object.
    """
    tree = mk_const(0.25018632)

    assert repr(tree) == "0.250186"
    assert "np.float32" not in repr(tree)
    assert isinstance(tree.to_sympy_expr(), sp.Basic)
    assert tree.to_infix() == "0.25"


@pytest.mark.skipif(
    shutil.which("dot") is None or importlib.util.find_spec("pygraphviz") is None,
    reason="to_png needs pygraphviz and the graphviz 'dot' binary",
)
def test_tree_to_png(small_tree, tmp_path):
    target = tmp_path / "tree.png"
    small_tree.to_png(str(target))
    assert target.exists() and target.stat().st_size > 0


# --------------------------------------------------------------------------- #
# execute_mode strings
# --------------------------------------------------------------------------- #
def collect_sr_fitness_modes(obj, inputs, labels, use_mse=True):
    """Call ``SR_fitness`` with every documented mode; required modes must work."""
    results = {}
    for mode in REQUIRED_SR_MODES + OPTIONAL_SR_MODES:
        try:
            value = obj.SR_fitness(inputs, labels, use_mse, mode)
        except Exception as exc:
            if mode in REQUIRED_SR_MODES:
                raise AssertionError(f"SR_fitness(execute_mode={mode!r}) was rejected: {exc!r}")
            continue
        results[mode] = value.detach().cpu().float()
    return results


def test_tree_sr_fitness_modes_agree(small_tree):
    inputs, labels = sr_data(small_tree)
    results = collect_sr_fitness_modes(small_tree, inputs, labels)
    reference = results["auto"]
    assert reference.shape == (1,), f"Tree.SR_fitness must return a (1,) tensor, got {reference.shape}"
    for mode, value in results.items():
        assert value.shape == reference.shape
        assert torch.allclose(value, reference, rtol=1e-3, atol=1e-4, equal_nan=True), (
            f"Tree.SR_fitness(execute_mode={mode!r}) disagrees with {results['auto']}"
        )


def test_forest_sr_fitness_modes_agree(small_population):
    inputs, labels = sr_data(small_population)
    results = collect_sr_fitness_modes(small_population, inputs, labels)
    reference = results["auto"]
    assert reference.shape == (small_population.pop_size,)
    for mode, value in results.items():
        assert value.shape == reference.shape
        assert torch.allclose(value, reference, rtol=1e-3, atol=1e-4, equal_nan=True), (
            f"Forest.SR_fitness(execute_mode={mode!r}) disagrees with "
            f"Forest.SR_fitness(execute_mode='auto')"
        )


def test_sr_fitness_matches_torch_mean_squared_error(small_population):
    """The fitness magnitude must be the MSE of ``batch_forward`` (sign convention
    is up to the caller, ``problem/symbolic_regression.py`` negates it)."""
    inputs, labels = sr_data(small_population)
    prediction = small_population.batch_forward(inputs).detach().cpu().float()
    mse = ((prediction - labels.cpu()[None, :, :]) ** 2).mean(dim=(1, 2))

    for mode, value in collect_sr_fitness_modes(small_population, inputs, labels).items():
        assert torch.allclose(
            value.abs(), mse, rtol=1e-3, atol=1e-4, equal_nan=True
        ), f"SR_fitness(execute_mode={mode!r}) is not the mean squared error"


def test_tree_and_forest_sr_fitness_agree(small_population):
    single = small_population[0:1]
    tree = small_population[0]
    inputs, labels = sr_data(tree)
    forest_value = single.SR_fitness(inputs, labels, True, "auto").detach().cpu().float()
    tree_value = tree.SR_fitness(inputs, labels, True, "auto").detach().cpu().float()
    assert float(forest_value[0]) == pytest.approx(float(tree_value[0]), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("unknown", ["nope", "AUTO", "torch parallel", ""])
def test_unknown_sr_fitness_execute_mode_raises(small_population, unknown):
    inputs, labels = sr_data(small_population)
    with pytest.raises(Exception):
        small_population.SR_fitness(inputs, labels, True, unknown)
    with pytest.raises(Exception):
        small_population[0].SR_fitness(inputs, labels, True, unknown)


def test_forest_forward_mode_strings_when_supported(small_population):
    if not uses_execute_mode(Forest.forward):
        pytest.skip("Forest.forward does not expose execute_mode in this revision")
    inputs = torch.rand(
        small_population.pop_size,
        small_population.input_len,
        device=small_population.batch_node_value.device,
    )
    results = {}
    for mode in FORWARD_MODES:
        try:
            results[mode] = small_population.forward(inputs, execute_mode=mode).detach().cpu()
        except Exception as exc:
            if mode == "auto":
                raise AssertionError(f"Forest.forward(execute_mode={mode!r}) failed: {exc!r}")
    assert results, "Forest.forward accepted no execute_mode string"
    reference = results.get("auto", next(iter(results.values())))
    for mode, value in results.items():
        assert torch.allclose(value.float(), reference.float(), rtol=1e-4, atol=1e-5, equal_nan=True)
    with pytest.raises(Exception):
        small_population.forward(inputs, execute_mode="not-a-mode")


def test_tree_forward_mode_strings_when_supported(small_tree):
    if not uses_execute_mode(Tree.forward):
        pytest.skip("Tree.forward does not expose execute_mode in this revision")
    point = torch.rand(
        1, small_tree.input_len, device=small_tree.node_value.device
    )
    values = []
    for mode in FORWARD_MODES:
        try:
            values.append(small_tree.forward(point, execute_mode=mode).detach().cpu().float())
        except Exception as exc:
            if mode == "auto":
                raise AssertionError(f"Tree.forward(execute_mode={mode!r}) failed: {exc!r}")
    assert values
    for value in values[1:]:
        assert torch.allclose(value, values[0], rtol=1e-4, atol=1e-5, equal_nan=True)
    with pytest.raises(Exception):
        small_tree.forward(point, execute_mode="not-a-mode")


def test_forest_batch_forward_matches_forward(small_population):
    n = small_population.pop_size
    inputs = torch.rand(n, small_population.input_len, device=small_population.batch_node_value.device)
    single = small_population.forward(inputs).detach().cpu().float()  # (pop, output)
    batched = small_population.batch_forward(inputs).detach().cpu().float()  # (pop, n, output)

    assert tuple(batched.shape) == (n, n, small_population.output_len)
    for i in range(n):
        # Forest.forward evaluates tree i at inputs[i]; batch_forward[i, j] is tree i at inputs[j]
        assert torch.allclose(single[i], batched[i, i], rtol=1e-4, atol=1e-5, equal_nan=True)


# --------------------------------------------------------------------------- #
# Module helpers and the descriptor surface
# --------------------------------------------------------------------------- #
def test_randint_smoke():
    values = randint((4, 3), 3, 9)
    assert values.dtype == torch.int32
    assert tuple(values.shape) == (4, 3)
    assert torch.all(values.cpu() >= 3) and torch.all(values.cpu() <= 9)

    zeros = randint(5, 0, 1)
    assert zeros.dtype == torch.int32
    assert torch.all((zeros.cpu() == 0) | (zeros.cpu() == 1))


def test_generate_descriptor_construction_surface():
    # first three parameters stay positional, the rest keyword
    positional = GenerateDescriptor(
        64, 2, 1, using_funcs=["+"], max_layer_cnt=3, const_range=(-1.0, 1.0), sample_cnt=4
    )
    assert (positional.max_tree_len, positional.input_len, positional.output_len) == (64, 2, 1)

    descriptor = GenerateDescriptor(
        max_tree_len=64,
        input_len=3,
        output_len=2,
        const_prob=0.25,
        out_prob=0.75,
        using_funcs=["+", "sin", "if"],
        max_layer_cnt=3,
        layer_leaf_prob=0.3,
        const_range=(-2.0, 2.0),
        sample_cnt=6,
    )
    assert (descriptor.const_prob, descriptor.out_prob) == (0.25, 0.75)
    assert descriptor.depth2leaf_probs.dim() == 1
    assert descriptor.roulette_funcs.shape == (NUM_FUNCS,)
    assert descriptor.const_samples.dim() == 1 and descriptor.const_samples.shape[0] == 6
    assert float(descriptor.const_samples.min()) >= -2.0
    assert float(descriptor.const_samples.max()) <= 2.0

    updated = descriptor.update(max_tree_len=128)
    assert isinstance(updated, GenerateDescriptor)
    assert updated.max_tree_len == 128
    assert (updated.input_len, updated.output_len) == (3, 2)
    assert updated.const_prob == 0.25


def test_generate_descriptor_accepts_explicit_tensor_arguments():
    """The precomputed-tensor path of the descriptor must keep working."""
    device = api_device()
    descriptor = GenerateDescriptor(
        max_tree_len=32,
        input_len=2,
        output_len=1,
        depth2leaf_probs=torch.tensor([0.5] * 4 + [1.0] * 6, device=device),
        # cumulative "roulette" vector over the 29 functions: all mass on "+" (id 1)
        roulette_funcs=torch.cumsum(
            torch.tensor([0.0, 1.0] + [0.0] * 27, device=device), dim=0
        ),
        const_samples=torch.tensor([0.5, -0.5], device=device),
    )
    assert descriptor.roulette_funcs.shape == (NUM_FUNCS,)
    forest = Forest.random_generate(pop_size=4, descriptor=descriptor)
    assert forest.pop_size == 4 and forest.max_tree_len == 32
