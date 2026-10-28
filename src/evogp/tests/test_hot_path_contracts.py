"""Contract tests for the operator / training-loop hot path.

These tests pin the *performance contracts* of the GP operator layer, not its
numerics (``test_operators.py`` / ``test_equivalence.py`` own those):

* ``tree/utils.check_tensor`` short-circuits a tensor that already lives on the
  engine device without an autograd graph (it used to run ``to``/``detach``/
  ``requires_grad_`` on every operator call);
* ``tree/generation.generate_population`` produces a valid prefix encoding on
  BOTH layouts -- the padded complete-tree hot path and the compact fallback --
  and ``PADDED_SLOT_BUDGET`` really selects between them;
* ``algorithm/mutation/default.DefaultMutation`` mutates exactly the rows of the
  ``torch.rand(pop) < rate`` draw, in place, and the operators as a whole do not
  sync with the host per individual;
* ``pipeline/standard.StandardPipeline.step`` maps ``NaN`` fitness to ``-inf``
  without touching ``+inf`` / ``-inf``.

Run from the repository root::

    PYTHONPATH=src:/root/evoxc /tmp/genesis/task_967a300e3bce7f35/evogp-venv/bin/python \
        -m pytest src/evogp/tests/test_hot_path_contracts.py -q
"""
from __future__ import annotations

import torch
import pytest

from evogp.algorithm import DefaultCrossover, DefaultMutation, DefaultSelection
from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.tree.utils import check_tensor
from evogp.tree import generation as generation_module
from evogp.tree import splicing

# arity indexed by base node type (VAR, CONST, UFUNC, BFUNC, TFUNC)
ARITY = (0, 0, 1, 2, 3)
TYPE_MASK = 0x7F

_DEVICE = None


def api_device() -> torch.device:
    """Device used by the engine's own tensor factories (never hard-coded twice)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = Forest.zero_generate(1, 4, 1, 1).batch_node_value.device
    return _DEVICE


def descriptor(**kwargs) -> GenerateDescriptor:
    params = dict(
        max_tree_len=32,
        input_len=3,
        output_len=1,
        const_prob=0.4,
        out_prob=0.0,
        using_funcs=["+", "-", "*"],
        max_layer_cnt=3,
        layer_leaf_prob=0.3,
        const_range=(-1.0, 1.0),
        sample_cnt=8,
    )
    params.update(kwargs)
    return GenerateDescriptor(**params)


def assert_valid_prefix(forest: Forest, label: str) -> None:
    """Recursively validate the prefix (pre-order) encoding of every row."""
    node_type = forest.batch_node_type.cpu().to(torch.int64).tolist()
    node_value = forest.batch_node_value.cpu().tolist()
    subtree_size = forest.batch_subtree_size.cpu().to(torch.int64).tolist()
    max_tree_len = forest.max_tree_len

    for row in range(forest.pop_size):
        total = subtree_size[row][0]
        assert 1 <= total <= max_tree_len, f"{label}: row {row} has length {total}"

        def walk(index: int) -> int:
            base = node_type[row][index] & TYPE_MASK
            assert base in (0, 1, 2, 3, 4), f"{label}: row {row} node {index} type {base}"
            cursor = index + 1
            for _ in range(ARITY[base]):
                cursor += walk(cursor)
            assert subtree_size[row][index] == cursor - index, (
                f"{label}: row {row} node {index} subtree size "
                f"{subtree_size[row][index]} != {cursor - index}"
            )
            return cursor - index

        walk(0)
        for column in range(total, max_tree_len):
            assert (node_value[row][column], node_type[row][column],
                    subtree_size[row][column]) == (0.0, 0, 0), (
                f"{label}: row {row} column {column} is not zero padding"
            )


# --------------------------------------------------------------------------- #
# check_tensor
# --------------------------------------------------------------------------- #
def test_check_tensor_short_circuits_a_ready_tensor():
    """An engine-device tensor without grad is passed through (same object)."""
    x = torch.zeros(4, dtype=torch.int32, device=api_device())
    assert check_tensor(x) is x


def test_check_tensor_still_relocates_and_detaches():
    """The documented contract of the slow path: ``"cuda"``, no grad, same values."""
    cpu = torch.arange(6, dtype=torch.float32)
    out = check_tensor(cpu)
    assert out.device.type == "cuda"
    assert out.requires_grad is False
    assert torch.equal(out.cpu(), cpu)

    grad = torch.ones(3, dtype=torch.float32, device=api_device(), requires_grad=True)
    detached = check_tensor(grad)
    assert detached.requires_grad is False
    assert detached is not grad


# --------------------------------------------------------------------------- #
# generation layouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("budget_name", ["padded", "compact"])
def test_generate_population_encodes_a_valid_prefix(monkeypatch, budget_name):
    """Both the padded hot path and the compact fallback emit valid prefixes."""
    if budget_name == "compact":
        monkeypatch.setattr(generation_module, "PADDED_SLOT_BUDGET", 0)

    for kwargs in (
        {},
        {"using_funcs": ["+", "sin", "if"], "max_layer_cnt": 3},
        {"using_funcs": ["sin", "neg"], "max_layer_cnt": 4},
        {"output_len": 3, "out_prob": 1.0, "max_layer_cnt": 2},
    ):
        torch.manual_seed(0)
        forest = Forest.random_generate(pop_size=24, descriptor=descriptor(**kwargs))
        assert_valid_prefix(forest, f"{budget_name}/{kwargs}")


def test_padded_layout_is_selected_for_a_binary_descriptor(monkeypatch):
    """Sanity check of the layout switch: binary funcs -> 7 slots, unary -> budget."""
    assert generation_module._padded_total(2, 3) == 7
    assert generation_module._padded_total(2, 3) <= generation_module.PADDED_SLOT_BUDGET

    torch.manual_seed(1)
    padded = Forest.random_generate(pop_size=12, descriptor=descriptor())
    monkeypatch.setattr(generation_module, "PADDED_SLOT_BUDGET", 0)
    torch.manual_seed(1)
    compact = Forest.random_generate(pop_size=12, descriptor=descriptor())

    # both layouts are valid, reproducible and (by construction) different streams
    assert_valid_prefix(padded, "padded")
    assert_valid_prefix(compact, "compact")
    torch.manual_seed(1)
    again = Forest.random_generate(pop_size=12, descriptor=descriptor())
    assert torch.equal(again.batch_subtree_size, compact.batch_subtree_size)


def test_cols_cache_is_shared_and_read_only():
    """The cached column index row is reused and never written through."""
    first = splicing._cols(64, api_device())
    second = splicing._cols(64, api_device())
    assert first is second
    assert first.dtype == torch.int64 and tuple(first.shape) == (1, 64)
    assert torch.equal(first, torch.arange(64, device=first.device).unsqueeze(0))


# --------------------------------------------------------------------------- #
# mutation / host round trips
# --------------------------------------------------------------------------- #
def test_default_mutation_uses_the_rand_mask_and_mutates_in_place():
    """Exactly the ``torch.rand(pop) < rate`` rows are rewritten, in place."""
    pop_size, rate, seed = 48, 0.3, 0
    desc = descriptor()

    # the operator draws its mask from the CPU generator as its very first draw,
    # so it can be replayed exactly after re-seeding
    torch.manual_seed(seed)
    expected_mask = torch.rand(pop_size) < rate

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    forest = Forest.random_generate(pop_size=pop_size, descriptor=desc)
    before = forest.batch_node_value.clone()
    operator = DefaultMutation(mutation_rate=rate, descriptor=desc)

    torch.manual_seed(seed)
    out = operator(forest)

    assert out is forest, "DefaultMutation must mutate the forest in place"
    unchanged = (out.batch_node_value == before).all(dim=1).cpu()
    assert bool(unchanged[~expected_mask].all()), "a row outside the mask was rewritten"
    assert int((~unchanged[expected_mask]).sum()) >= int(expected_mask.sum()) - 1, (
        "the rows inside the mask were not replaced"
    )


class _HostTripCounter:
    """Count python-level host round trips on tensors (``.item``/``bool``/...)."""

    NAMES = ("item", "tolist", "cpu", "__int__", "__float__", "__bool__")

    def __init__(self):
        self.counts = {name: 0 for name in self.NAMES}
        self._saved = {}

    def __enter__(self):
        for name in self.NAMES:
            original = getattr(torch.Tensor, name)

            def wrapper(*args, _original=original, _name=name, **kwargs):
                self.counts[_name] += 1
                return _original(*args, **kwargs)

            self._saved[name] = original
            setattr(torch.Tensor, name, wrapper)
        return self

    def __exit__(self, *exc_info):
        for name, original in self._saved.items():
            setattr(torch.Tensor, name, original)
        return False

    @property
    def total(self):
        return sum(self.counts.values())


def test_operators_do_not_sync_per_individual():
    """One generation of the default operators syncs a handful of times, not per row."""
    pop_size, seed = 32, 0
    desc = descriptor()
    torch.manual_seed(seed)
    forest = Forest.random_generate(pop_size=pop_size, descriptor=desc)
    fitness = torch.rand(pop_size, device=api_device())
    selection = DefaultSelection(survival_rate=0.3, elite_rate=0.1)
    crossover = DefaultCrossover()
    mutation = DefaultMutation(mutation_rate=0.3, descriptor=desc)

    elites, survivors = selection(forest, fitness)
    with _HostTripCounter() as counter:
        offspring = crossover(
            forest=forest,
            survivor_indices=survivors,
            target_cnt=pop_size - len(elites),
            fitness=fitness,
        )
        offspring = mutation(offspring)
        # the elite splice of GeneticProgramming.step
        next_forest = forest[elites] + offspring

    assert next_forest.pop_size == pop_size
    # documented budget: the crossover range check (1), the generation overflow
    # guard (1) and the mutation's empty-selection test (0, it uses numel) --
    # anything per individual would be ~pop_size * 3
    assert counter.total <= 4, f"too many host round trips per generation: {counter.counts}"


# --------------------------------------------------------------------------- #
# pipeline fitness handling
# --------------------------------------------------------------------------- #
class _StubAlgorithm:
    """Minimal ``GeneticProgramming`` stand-in: the pipeline only needs these two."""

    def __init__(self, forest: Forest):
        self.forest = forest
        self.seen = None

    def step(self, fitness):
        self.seen = fitness.detach().clone()
        return self.forest


class _FixedFitnessProblem:
    """Problem returning a fixed fitness vector (NaN / +inf / -inf included)."""

    def __init__(self, values):
        self.values = values

    def evaluate(self, forest):
        return torch.tensor(self.values, dtype=torch.float32, device=api_device())


def test_pipeline_step_maps_nan_to_neg_inf_and_keeps_infinities():
    values = [float("nan"), float("inf"), float("-inf"), -0.5, 0.25]
    forest = Forest.zero_generate(len(values), 4, 1, 1)
    algorithm = _StubAlgorithm(forest)
    pipeline = StandardPipeline(
        algorithm, _FixedFitnessProblem(values), generation_limit=1, is_show_details=False
    )

    cpu_fitness = pipeline.step()

    assert cpu_fitness.device.type == "cpu"
    assert float(cpu_fitness[0]) == float("-inf"), "NaN fitness must become -inf"
    assert float(cpu_fitness[1]) == float("inf"), "+inf must not be clamped"
    assert float(cpu_fitness[2]) == float("-inf"), "-inf must not be clamped"
    assert torch.equal(algorithm.seen.cpu(), cpu_fitness)
    assert pipeline.best_fitness == float("inf")
    assert pipeline.best_tree is not None
