"""The single multi-output (``OUT_NODE``) contract rule, and the engines that apply it.

A node type tagged with ``NType.OUT_NODE`` (bit 7) belongs to a population built
for ``output_len >= 2``; under ``output_len == 1`` the legacy kernels did not mask
the bit at all, so a tagged node was evaluated as a TERNARY ``if`` node with its
packed payload read as a variable index -- a plausible wrong number, no error.
That is the one failure mode nobody can debug, so the rule lives in ONE place
(``evogp.tree.multi_output.check_multi_output``) and every engine refuses through
it, on the host, before a decode / compile / launch:

====================== ==================================================
site                   what it raises
====================== ==================================================
``multi_output``       ``MultiOutputError`` (a ``ValueError``), ``DenseUnsupported``
                       when the caller passes ``error=``
``dense_engine``       ``DenseUnsupported`` (a DECLINE, so the route falls back)
``evoxc_engine``       ``MultiOutputError`` naming the row and the node
``fused_engine``       ``MultiOutputError``, before any launch (device-gated file
                       ``test_fused_route.py`` pins the launch behaviour)
``evogp.gp``           ``graph_eval.MultiOutputError`` (the torch-free mirror,
                       pinned in ``gp/tests/test_gp_stack_multi_output.py``)
====================== ==================================================

This module pins the rule itself and the two host-side engines, plus the two
sides that must NOT change: a genuine multi-output population (``output_len``
2/3) still evaluates, and a valid single-output population still equals the
plain-torch oracle (``reference_eval.py``) bit for bit on the arithmetic and
ternary surfaces.  ``conftest.py``'s ``EVOGP_EVOXC_BACKEND=numpy`` pin is the
whole route here, so nothing compiles a device graph and the module is hermetic.

Run from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \\
        <venv>/bin/python -m pytest src/evogp/tests/test_multi_output_contract.py -q
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, List

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

#: The shared fixtures (tiny spec DSL + deterministic row materializer + oracle)
#: are reused, guarded by ``ImportError`` only: a missing dependency skips, an
#: import REGRESSION inside them fails collection loudly.
try:
    import test_equivalence as eq
except ImportError as exc:  # pragma: no cover - only in a pre-merge worktree
    eq = None
    _FIXTURE_IMPORT_ERROR: Any = exc
else:
    _FIXTURE_IMPORT_ERROR = None

try:
    from evogp.tree import Forest
    from evogp.tree import dense_engine as D
    from evogp.tree import evoxc_engine as E
    from evogp.tree import fused_engine
    from evogp.tree import multi_output
    from evogp.tree.utils import NType
    _ENGINE_IMPORT_ERROR: Any = None
except ImportError as exc:  # pragma: no cover - only when a dependency is missing
    Forest = D = E = fused_engine = multi_output = NType = None
    _ENGINE_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _ENGINE_IMPORT_ERROR is not None or _FIXTURE_IMPORT_ERROR is not None,
    reason=(
        "the multi-output contract needs evogp.tree + the shared test fixtures "
        f"(engine: {_ENGINE_IMPORT_ERROR!r}, fixtures: {_FIXTURE_IMPORT_ERROR!r})"
    ),
)

#: Test tensors follow the suite's convention (cuda when present); every engine
#: here is device-agnostic, so a CPU-only box runs the same assertions.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_LEN = 3
MAX_TREE_LEN = 16
ROWS = 12
OUT_NODE = int(NType.OUT_NODE)


# --------------------------------------------------------------------------- #
# helpers: populations, the oracle, a NaN/inf-exact comparator
# --------------------------------------------------------------------------- #
def _forest(specs: List[Any], *, output_len: int = 1, pad_to: int = MAX_TREE_LEN,
            input_len: int = INPUT_LEN) -> "Forest":
    """A device ``Forest`` from nested specs (the shared materializer)."""
    values, types, sizes = eq.rows_from_specs(specs, output_len=output_len, pad_to=pad_to)
    return Forest(
        input_len,
        output_len,
        values.to(DEVICE),
        types.to(DEVICE),
        sizes.to(DEVICE),
    )


def _data(rows: int, columns: int, seed: int = 5):
    """Deterministic positive data (LOG/SQRT/DIV stay finite and NaN-free)."""
    values = torch.rand(rows, columns, generator=torch.Generator().manual_seed(seed))
    return (values + 0.25).to(DEVICE)


def _tagged_specs() -> List[Any]:
    """A population built for ``output_len = 2`` (every tag is a real out-node)."""
    return [
        eq.fn(
            eq.ADD,
            eq.out_fn(eq.ADD, 0, eq.var(0), eq.var(1)),
            eq.out_fn(eq.MUL, 1, eq.var(1), eq.var(2)),
        ),
        eq.fn(
            eq.IF,
            eq.var(0),
            eq.out_fn(eq.TANH, 1, eq.var(1)),
            eq.out_fn(eq.ABS, 0, eq.var(2)),
        ),
    ]


def _single_output_specs() -> List[Any]:
    """A valid single-output population on the bitwise (arithmetic + ternary) surface."""
    shared = eq.fn(eq.ADD, eq.fn(eq.MUL, eq.var(0), eq.var(1)), eq.var(2))
    return [
        shared,
        eq.fn(eq.IF, eq.var(0), eq.var(1), eq.var(2)),
        eq.fn(eq.DIV, eq.fn(eq.SUB, eq.var(1), eq.var(2)), eq.fn(eq.ADD, eq.var(2), eq.const(1.5))),
        eq.const(2.5),
        shared,  # duplicate PROGRAM: one compile serves both trees
    ]


def _rows(forest: "Forest"):
    """The population as the oracle's CPU rows."""
    return (
        forest.batch_node_value.detach().cpu().float(),
        forest.batch_node_type.detach().cpu(),
        forest.batch_subtree_size.detach().cpu(),
    )


def _oracle(
    forest: "Forest",
    x: torch.Tensor,
    *,
    labels=None,
    use_mse: bool = True,
    per_row: bool = False,
):
    """The independent plain-torch answer for the same tensors."""
    values, types, sizes = _rows(forest)
    x_cpu = x.detach().cpu().float()
    if labels is None:
        emit = (
            eq.reference.forward_from_rows
            if per_row
            else eq.reference.batch_forward_from_rows
        )
        return emit(values, types, sizes, x_cpu, out_len=forest.output_len)
    return eq.reference.sr_fitness_from_rows(
        values,
        types,
        sizes,
        x_cpu,
        labels.detach().cpu().float(),
        use_mse=use_mse,
        out_len=forest.output_len,
    )


def _assert_bitwise(got, expected, context: str) -> None:
    """Bit-identical finite values, with NaN / ±inf placement compared exactly."""
    got = torch.as_tensor(got, dtype=torch.float32).cpu()
    want = torch.as_tensor(expected, dtype=torch.float32).cpu()
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
    assert torch.equal(
        got[finite].contiguous().view(torch.int32),
        want[finite].contiguous().view(torch.int32),
    ), f"{context}: the finite values are not bit-identical"


def _assert_loss_close(got, expected, context: str) -> None:
    """Losses ``~1e-7`` relative (the documented engine deviation), not bitwise."""
    got = torch.as_tensor(got, dtype=torch.float32).cpu()
    want = torch.as_tensor(expected, dtype=torch.float32).cpu()
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
    finite = torch.isfinite(got) & torch.isfinite(want)
    if not bool(finite.any()):
        return
    assert torch.allclose(got[finite], want[finite], rtol=1e-6, atol=1e-6), (
        f"{context}: {got[finite].tolist()} != {want[finite].tolist()}"
    )

def _pop_tables(specs: List[Any], *, output_len: int = 1, pad_to: int = MAX_TREE_LEN):
    """CPU numpy population tables of ``specs`` (for the rule-level tests)."""
    values, types, sizes = eq.rows_from_specs(specs, output_len=output_len, pad_to=pad_to)
    return values.numpy(), types.numpy(), sizes.numpy()


# --------------------------------------------------------------------------- #
# 1. the rule itself
# --------------------------------------------------------------------------- #
def test_max_out_len_is_single_sourced_in_the_rule_module():
    """``fused_engine.MAX_OUT_LEN`` is the rule module's value, not a second copy."""
    assert multi_output.MAX_OUT_LEN == 8
    assert fused_engine.MAX_OUT_LEN == multi_output.MAX_OUT_LEN


def test_find_tagged_node_reads_the_first_tagged_live_node():
    """Row-major prefix order, and the raw type comes back for the message."""
    types = np.zeros((3, 8), dtype=np.int16)
    types[1, 2] = np.int16(int(NType.BFUNC) | OUT_NODE)
    types[2, 0] = np.int16(int(NType.UFUNC) | OUT_NODE)
    assert multi_output.find_tagged_node(types) == (1, 2, int(NType.BFUNC) | OUT_NODE)
    # a shorter first row changes which tag is "first"
    assert multi_output.find_tagged_node(types, row_lengths=[1, 8, 1]) == (
        1,
        2,
        int(NType.BFUNC) | OUT_NODE,
    )


def test_find_tagged_node_ignores_the_padding_behind_a_tree():
    """Garbage padding may carry bit 7; the live mask is what the kernels walk."""
    types = np.full((2, 8), -1, dtype=np.int16)  # -1 carries bit 7 as well
    types[:, 0] = np.int16(int(NType.VAR))
    assert multi_output.find_tagged_node(types, row_lengths=[1, 1]) is None
    assert multi_output.find_tagged_node(types) == (0, 1, -1)  # unmasked = every cell


def test_find_tagged_node_accepts_a_single_row():
    """The 1-D form is what the specialized decoder's per-row loop passes."""
    row = np.zeros(4, dtype=np.int32)
    row[3] = np.int32(int(NType.TFUNC) | OUT_NODE)
    assert multi_output.find_tagged_node(row, row=7) == (7, 3, int(NType.TFUNC) | OUT_NODE)


def test_find_tagged_node_rejects_a_table_it_cannot_read():
    """A float table is a caller bug, not a population; a bad mask is a wiring bug."""
    with pytest.raises(TypeError, match="integer"):
        multi_output.find_tagged_node(np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="row_lengths"):
        multi_output.find_tagged_node(
            np.zeros((2, 4), dtype=np.int16), row_lengths=[4]
        )
    with pytest.raises(ValueError, match="1-D"):
        multi_output.find_tagged_node(np.zeros((1, 2, 4), dtype=np.int16))


def test_check_multi_output_raises_the_shared_error_for_a_single_output_request():
    """The default error is the dual ``ValueError`` / ``NotImplementedError`` class."""
    _, types, sizes = _pop_tables(_tagged_specs(), output_len=2)
    assert multi_output.find_tagged_node(types, sizes[:, 0]) is not None
    with pytest.raises(multi_output.MultiOutputError) as caught:
        multi_output.check_multi_output(types, 1, row_lengths=sizes[:, 0])
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, NotImplementedError)
    message = str(caught.value)
    assert "row 0: node 1" in message, message
    assert "OUT_NODE" in message and "output_len == 1" in message, message


def test_check_multi_output_accepts_the_multi_output_protocol():
    """``output_len >= 2`` makes the tag the node's output index: nothing to refuse."""
    _, types, sizes = _pop_tables(_tagged_specs(), output_len=2)
    for output_len in (2, 3):
        assert multi_output.check_multi_output(types, output_len, row_lengths=sizes[:, 0]) is None


def test_check_multi_output_honours_the_caller_s_error_type():
    """The dense route DECLINES (its own exception) instead of raising."""
    _, types, sizes = _pop_tables(_tagged_specs(), output_len=2)
    with pytest.raises(D.DenseUnsupported, match="OUT_NODE"):
        multi_output.check_multi_output(types, 1, row_lengths=sizes[:, 0], error=D.DenseUnsupported)
    with pytest.raises(ValueError, match="output_len"):
        multi_output.check_multi_output(types, 0)


def test_a_clean_population_passes_the_rule():
    """No tag anywhere: the rule is silent, so no numeric path can change."""
    _, types, sizes = _pop_tables(_single_output_specs())
    assert multi_output.find_tagged_node(types, sizes[:, 0]) is None
    assert multi_output.check_multi_output(types, 1, row_lengths=sizes[:, 0]) is None


# --------------------------------------------------------------------------- #
# 2. the two host-side engines refuse through the rule
# --------------------------------------------------------------------------- #
def _tagged_forest() -> "Forest":
    """A population BUILT for ``output_len = 2``, handed to the engine as single-output.

    The mis-wired container: the rows carry real out-nodes (the tags are the
    out-nodes' own), while ``Forest.output_len`` says 1 -- exactly what a wrong
    ``GenerateDescriptor.output_len`` produces.
    """
    values, types, sizes = eq.rows_from_specs(
        _tagged_specs(), output_len=2, pad_to=MAX_TREE_LEN
    )
    return Forest(INPUT_LEN, 1, values.to(DEVICE), types.to(DEVICE), sizes.to(DEVICE))


def test_specialized_decoder_refuses_a_tagged_single_output_population():
    """``decode_population`` owns the malformed-tree ``ValueError`` contract."""
    forest = _tagged_forest()
    assert bool((forest.batch_node_type & OUT_NODE).any()), "the fixture lost its tags"
    with pytest.raises(ValueError, match="OUT_NODE") as caught:
        E.decode_population(
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
            forest.input_len,
            1,
        )
    assert isinstance(caught.value, multi_output.MultiOutputError), (
        "the decoder must refuse through the shared rule, not a private copy"
    )
    assert "row 0: node 1" in str(caught.value), str(caught.value)


def test_dense_engine_declines_a_tagged_single_output_population():
    """The dense route DECLINES (it must keep falling back), naming the same rule."""
    forest = _tagged_forest()
    with pytest.raises(D.DenseUnsupported, match="OUT_NODE"):
        D.prepare_dense(
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
            forest.input_len,
            1,
        )


@pytest.mark.parametrize("mode", ["specialized", "dense", "auto"])
def test_the_public_single_output_path_refuses(mode, monkeypatch):
    """Every route ends on a ``ValueError`` -- never on a plausible prediction."""
    monkeypatch.setenv("EVOGP_EVAL_MODE", mode)
    forest = _tagged_forest()
    x = _data(ROWS, INPUT_LEN)
    with pytest.raises(ValueError, match="OUT_NODE"):
        forest.batch_forward(x)
    with pytest.raises(ValueError, match="OUT_NODE"):
        forest.SR_fitness(x, _data(ROWS, 1, seed=11))


def test_a_valid_single_output_population_is_untouched():
    """The refusal must be invisible to input that was always legal: bitwise."""
    forest = _forest(_single_output_specs())
    x, labels = _data(ROWS, INPUT_LEN), _data(ROWS, 1, seed=13)
    _assert_bitwise(forest.batch_forward(x), _oracle(forest, x), "batch_forward[valid]")
    _assert_loss_close(
        forest.SR_fitness(x, labels), _oracle(forest, x, labels=labels), "SR_fitness[valid]"
    )
    per_row = _data(forest.pop_size, INPUT_LEN, seed=17)
    _assert_bitwise(
        forest.forward(per_row),
        _oracle(forest, per_row, per_row=True),
        "forward[valid]",
    )


# --------------------------------------------------------------------------- #
# 3. a genuine multi-output population still evaluates, bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("output_len", [2, 3])
def test_a_multi_output_population_still_evaluates_bitwise(output_len):
    """``output_len`` 2/3 on the arithmetic+ternary surface: same numbers as before."""
    forest = _forest(_tagged_specs(), output_len=output_len, pad_to=MAX_TREE_LEN)
    x, labels = _data(ROWS, INPUT_LEN), _data(ROWS, output_len, seed=19)
    numbers = (NType.OUT_NODE, NType.TYPE_MASK)
    assert numbers == (128, 0x7F), "the tag / mask the fixtures encode against"
    _assert_bitwise(
        forest.batch_forward(x),
        _oracle(forest, x),
        f"batch_forward[multi, out_len={output_len}]",
    )
    per_row = _data(forest.pop_size, INPUT_LEN, seed=23)
    _assert_bitwise(
        forest.forward(per_row),
        _oracle(forest, per_row, per_row=True),
        f"forward[multi, out_len={output_len}]",
    )
    # the loss path reduces over the DATA rows, and summation order alone costs
    # ~1 ulp (the deviation `VALIDATION.md` records for every route), so the loss
    # is compared at the documented ~1e-7 relative level rather than bitwise.
    _assert_loss_close(
        forest.SR_fitness(x, labels),
        _oracle(forest, x, labels=labels),
        f"SR_fitness[multi, out_len={output_len}]",
    )


def test_the_decoder_still_accepts_a_tagged_population_at_output_len_two():
    """The refusal is scoped to ``output_len == 1`` -- the decode itself is unchanged."""
    values, types, sizes = eq.rows_from_specs(_tagged_specs(), output_len=2, pad_to=MAX_TREE_LEN)
    population = E.decode_population(values, types, sizes, INPUT_LEN, 2)
    assert population.output_len == 2
    assert population.pop_size == len(_tagged_specs())
    x = _data(ROWS, INPUT_LEN)
    got = E.evaluate(population, x, per_row=False)
    want = eq.reference.batch_forward_from_rows(
        values, types, sizes, x.detach().cpu().float(), out_len=2
    )
    _assert_bitwise(got, want, "specialized evaluate[multi, out_len=2]")
