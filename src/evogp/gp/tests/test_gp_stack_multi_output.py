"""The multi-output (``OUT_NODE``) refusal at the ``graph_eval`` boundary.

``evogp.gp`` implements the SINGLE-output protocol only, and every other door into
the stack already says so: ``evaluator.predict`` (via ``_prepare_channels``),
``encoding.validate_channels`` / ``encode_population`` and
``fused_eval.FusedDeviceEvaluator`` all reject an ``OUT_NODE``-tagged opcode
channel.  ``graph_eval`` -- the DEVICE-RESIDENT route, and the one
``GraphEvaluator`` plugs into a driver loop -- used to trace the tagged cell as a
plain function id and return a plausible wrong fitness.  It now raises the same
refusal, from the single place both of its entry points share
(``_channel_triple``), so ``sr_fitness_channels``,
``sr_fitness_channels_numpy`` and ``GraphEvaluator.evaluate`` cannot disagree.

What is pinned here:

1. a tagged population raises ``MultiOutputError`` on all three entry points, and
   that class is both a ``ValueError`` and a ``NotImplementedError`` (the two
   spellings callers of the sibling guards already catch);
2. the refusal names the row and the node, and it fires for ANY tagged cell, not
   only the root;
3. the sibling host evaluator refuses the same population -- the rule is one
   contract, not one code path;
4. an UNTAGGED population is untouched: ``sr_fitness_channels_numpy`` and
   ``GraphEvaluator`` stay bit-identical to ``SymbolicRegression.evaluate``.

The check is HOST-SIDE by design: ``_host_opcode`` skips a device payload so the
route never pays a readback it exists to avoid (see the module docstring of
``evogp/gp/graph_eval.py`` and ``evogp/gp/CONTEXT.md``).  That limit is worth a
test of its own, and it is the last case here.

Row widths and populations are deliberately tiny: this suite is not a benchmark,
and the numpy backend builds these graphs in milliseconds.
"""

import numpy as np
import pytest

from evogp.gp import GPTreeConfig, init_population
from evogp.gp import graph_eval
from evogp.gp._constants import NType
from evogp.gp.evaluator import predict
from evogp.gp.graph_eval import (
    GraphEvaluator,
    MultiOutputError,
    sr_fitness_channels,
    sr_fitness_channels_numpy,
)
from evogp.gp.problem import SymbolicRegression

#: ``(name, functions, max_layer_cnt, max_nodes, pop_size, input_len, n_rows, seed)``
#: -- two surfaces, so the refusal is not an artefact of one function set.
CASES = [
    ("arith", ["+", "-", "*", "/"], 3, 16, 6, 3, 5, 31),
    ("if-cmp", ["if", "<", ">", "+", "-", "max", "min"], 3, 16, 5, 2, 4, 32),
]


def _case(case):
    """``(name, state, channels, x, y)`` -- numpy leaves, so every door is host-side."""
    name, funcs, depth, max_nodes, pop, input_len, n_rows, seed = case
    config = GPTreeConfig(
        max_nodes=max_nodes,
        input_len=input_len,
        using_funcs=funcs,
        max_layer_cnt=depth,
        layer_leaf_prob=0.25,
        const_samples=(0.5, -1.0, 2.0, 0.0),
    )
    state = init_population(config, seed, pop_size=pop)
    channels = tuple(np.asarray(leaf.numpy()) for leaf in state.channels())
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_rows, input_len)).astype(np.float32)
    x[0, 0] = 0.0  # the zero edges: div / log / pow(0, 0)
    y = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    return name, state, channels, x, y


def _tagged(channels, row: int, node: int):
    """``channels`` with the ``OUT_NODE`` bit set on ``(row, node)``, plus that cell."""
    opcode, payload, size = (np.array(leaf, copy=True) for leaf in channels)
    if int(size[row, node]) == 0:
        raise ValueError(
            f"({row}, {node}) is padding, not a node -- pick a live cell"
        )
    opcode[row, node] = int(opcode[row, node]) | int(NType.OUT_NODE)
    return (opcode, payload, size)


# --------------------------------------------------------------------------- #
# 1. every graph_eval entry point refuses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_graph_eval_refuses_a_tagged_population(case):
    """``sr_fitness_channels``, its numpy twin and ``GraphEvaluator`` all raise."""
    name, state, channels, x, y = _case(case)
    tagged = _tagged(channels, 0, 1)

    with pytest.raises(MultiOutputError) as caught:
        sr_fitness_channels(tagged, x, y)
    assert isinstance(caught.value, ValueError), (
        "the tree engines answer an undecodable population with ValueError"
    )
    assert isinstance(caught.value, NotImplementedError), (
        "this stack's own sibling guards (encoding/evaluator/fused_eval) raise "
        "NotImplementedError for the protocol they do not implement"
    )
    assert "OUT_NODE" in str(caught.value), str(caught.value)
    assert "row 0: node 1" in str(caught.value), str(caught.value)

    with pytest.raises(MultiOutputError, match="OUT_NODE"):
        sr_fitness_channels_numpy(tagged, x, y)

    evaluator = GraphEvaluator(x, y, used_ids=None)
    with pytest.raises(MultiOutputError, match="OUT_NODE"):
        evaluator.evaluate(tagged)
    with pytest.raises(MultiOutputError, match="OUT_NODE"):
        evaluator.evaluate_numpy(tagged)
    with pytest.raises(MultiOutputError, match="OUT_NODE"):
        evaluator(tagged)


@pytest.mark.parametrize(
    "row_node", [(0, 0), (0, 1), ("last", 0)], ids=lambda value: repr(value)
)
def test_the_refusal_names_the_first_tagged_live_cell(row_node):
    """The whole table is scanned, and the message points at the offending cell."""
    _, _, channels, _, _ = _case(CASES[1])
    row, node = row_node
    size = channels[2]
    if row == "last":
        # the deepest live cell of the widest row: the tail of the population
        row = int(np.argmax(size[:, 0]))
        node = int(size[row, 0]) - 1
    tagged = _tagged(channels, row, node)
    with pytest.raises(MultiOutputError) as caught:
        sr_fitness_channels_numpy(tagged, channels[0] * 0, channels[0][:, 0] * 0)
    assert f"row {row}: node {node}" in str(caught.value), str(caught.value)


def test_the_sibling_host_evaluator_refuses_the_same_population():
    """One contract, two doors: ``evaluator.predict`` raises for the same input."""
    _, state, channels, x, _ = _case(CASES[0])
    tagged = _tagged(channels, 0, 1)
    with pytest.raises(NotImplementedError, match="OUT_NODE"):
        predict(tagged, x)
    with pytest.raises(NotImplementedError, match="OUT_NODE"):
        sr_fitness_channels_numpy(tagged, x, np.zeros(x.shape[0], dtype=np.float32))


def test_the_refusal_is_scoped_to_the_tag():
    """An untagged population with the SAME structure is still evaluated."""
    _, _, channels, x, y = _case(CASES[0])
    clean = sr_fitness_channels_numpy(channels, x, y)
    assert clean.shape == (channels[0].shape[0],)
    assert np.isfinite(clean).all(), "the arithmetic case must stay finite"


# --------------------------------------------------------------------------- #
# 2. the valid single-output path is untouched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_the_valid_single_output_path_is_unchanged(case, use_mse):
    """Bit-identical to ``SymbolicRegression.evaluate`` -- the refusal costs nothing."""
    name, state, channels, x, y = _case(case)
    reference = SymbolicRegression(x, y, use_mse=use_mse).evaluate(state)
    got = sr_fitness_channels_numpy(channels, x, y, use_mse=use_mse)
    evaluator = GraphEvaluator(x, y, use_mse=use_mse, used_ids=None)
    through_evaluator = evaluator.evaluate_numpy(channels)
    for label, value in (("sr_fitness_channels_numpy", got), ("GraphEvaluator", through_evaluator)):
        assert value.shape == reference.shape and value.dtype == np.float32
        np.testing.assert_array_equal(
            np.isneginf(value),
            np.isneginf(reference),
            err_msg=f"{name}/{label}: -inf pattern (use_mse={use_mse})",
        )
        finite = np.isfinite(reference)
        np.testing.assert_array_equal(
            value[finite],
            reference[finite],
            err_msg=f"{name}/{label}: finite fitness (use_mse={use_mse})",
        )


# --------------------------------------------------------------------------- #
# 3. the documented limit: a device payload is not read back to check
# --------------------------------------------------------------------------- #
class _DevicePayload:
    """A stand-in ``evoxc`` tensor whose payload lives on a NON-cpu device."""

    device = type("Device", (), {"kind": "rocm", "index": 0})()

    def numpy(self):  # pragma: no cover - must never be reached
        raise AssertionError("a device payload must not be read back for the check")


def test_a_device_payload_is_not_read_back_for_the_check():
    """The check is host-side by contract: it never forces a device-to-host copy.

    ``evogp/gp/graph_eval.py`` skips a non-cpu payload rather than pay the
    readback the route exists to avoid -- so on the DEVICE path the refusal is
    inherited from the population's producer (``encoding`` / ``variation``),
    which rejects ``output_len > 1``.  Documented here so the gap cannot be
    mistaken for coverage.
    """
    assert graph_eval._host_opcode(_DevicePayload()) is None
    host = np.zeros((2, 4), dtype=np.int32)
    assert graph_eval._host_opcode(host) is host
