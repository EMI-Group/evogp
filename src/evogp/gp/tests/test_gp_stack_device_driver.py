"""Driver tests: the DEVICE-RESIDENT run (``init_gp`` / ``step`` / ``run``).
``workflow`` drives both routes with the same functions: a population of host
arrays with a numpy objective is the loop the rest of the suite pins, and a
population of ``evoxc`` tensors ON A DEVICE with a device objective
(:class:`evogp.gp.graph_eval.GraphEvaluator`) is the device loop added here.
What this module pins, in order:

* ``init_population(..., backend="iree", device=<device>)`` (and ``init_gp``,
  which forwards to it) returns a state whose every leaf (channels, fitness,
  key) lives on that device, and whose generation-0 population is bit-identical
  to the host run's for the same seed -- the encoding graph is the same program
  on both routes;
* ``variation`` on a device state -- offspring-only, with ``elite_count`` and
  with an explicit ``elite_index`` -- reproduces the numpy-backend generation
  BITWISE (int channels and float payloads; measured, not assumed), the next
  key included, so the device path is a transport change and never a semantic
  one;
* the in-graph elite ranking IS the host rule: ``np.argsort(-fitness,
  kind="stable")``, i.e. the lowest row index wins a tie (a ``descending`` sort
  would flip that order), and a NaN is sanitized to ``-inf`` before it can win;
* one device ``step`` (``GraphEvaluator`` as ``evaluate``) produces the SAME
  population as the numpy ``step`` (int channels and float payloads bitwise),
  the elite rows the parent's own top-``elite_count`` rows, the offspring marked
  ``-inf`` -- and costs TWO graph runs: **zero** host readbacks and zero channel
  downloads, neither the fitness vector nor a channel leaves the device;
* ``run`` still reports: exactly ONE fitness readback per evaluated generation
  (the history and the best tree are built from it), plus the documented
  generation-0 validation and final ``best_channels`` downloads -- both droppable
  with ``validate=False`` -- and the device trajectory matches the numpy one
  inside 1e-5 while ``result.state`` stays device-resident;
* the device objective itself: ``GraphEvaluator.evaluate`` returns a device
  tensor (no sync), matches the host ``SymbolicRegression.evaluate`` inside 1e-5
  relative, and its numpy-backend twin is bitwise exact;
* the package surface exposes the device pieces (``GraphEvaluator``,
  ``sr_fitness_channels``, ``readback_fitness``, ``FusedEvaluator``,
  ``fused_available``) and the cache surface covers the device layer.

Everything is gated on a genuinely usable device: the module skips (never
fails) on a box without ``iree`` or without a visible ``rocm``/``cuda`` device,
and the tests pass ``backend="iree"`` + an explicit ``evoxc.Device`` to every
call because the suite's ``conftest.py`` pins the numpy backend for hermeticity.
"""

from __future__ import annotations

import numpy as np
import pytest

evoxc = pytest.importorskip(
    "evoxc", reason="the device driver is built on evoxc"
)

from evogp.gp import (
    GPConfig,
    GPTreeConfig,
    GraphEvaluator,
    SymbolicRegression,
    cache_stats,
    clear_cache,
    init_gp,
    init_population,
    run,
    smoke,
    step,
)
from evogp.gp import encoding, graph_eval
from evogp.gp import variation as V

from gp_operator_support import (
    DEVICE_SKIP_REASON,
    is_device_tensor,
    to_host,
    usable_device_kind,
)

# --------------------------------------------------------------------------- #
# availability gate: a real device behind the iree backend, or skip
# --------------------------------------------------------------------------- #
DEVICE_KIND = usable_device_kind()

pytestmark = pytest.mark.skipif(DEVICE_KIND is None, reason=DEVICE_SKIP_REASON)

#: The run this module pins.  Small on purpose (one compile per graph) and
#: fixed, so the cross-route comparisons below are deterministic.
SEED = 0
NUM_DATA = 64
POP_SIZE = 16
ELITE_COUNT = 2
TOL = 1e-6


# --------------------------------------------------------------------------- #
# fixtures (module scoped: one compile per graph for the whole module)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    """The device under test (an ``evoxc.Device``, never a bare string)."""
    return evoxc.Device(DEVICE_KIND, 0)


@pytest.fixture(scope="module")
def tree_config():
    return GPTreeConfig(
        max_nodes=16,
        input_len=2,
        using_funcs=("+", "-", "*", "/"),
        max_layer_cnt=4,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
    )


@pytest.fixture(scope="module")
def run_config(tree_config):
    return (tree_config, GPConfig(pop_size=POP_SIZE, elite_count=ELITE_COUNT))


@pytest.fixture(scope="module")
def data():
    """``(x, y)`` of the smoke SR problem, the same data both routes evaluate."""
    return smoke.make_data(NUM_DATA, seed=SEED)


@pytest.fixture(scope="module")
def host_state(run_config):
    """Generation 0 of the host (numpy) run."""
    return init_gp(run_config, SEED)


@pytest.fixture(scope="module")
def device_state(run_config, device):
    """Generation 0 of the device run -- the same seed, hence the same trees."""
    return init_gp(run_config, SEED, backend="iree", device=device)


@pytest.fixture(scope="module")
def evaluator(data, tree_config, device):
    """The device objective: in-graph SR fitness, device tensor out, no sync."""
    x, y = data
    return GraphEvaluator(x, y, used_ids=tree_config, backend="iree", device=device)


# --------------------------------------------------------------------------- #
# init: the population starts on the device
# --------------------------------------------------------------------------- #
def test_init_gp_puts_every_leaf_of_the_state_on_the_device(
    device_state, host_state, device
):
    for name in ("opcode", "payload", "size", "fitness", "key"):
        leaf = getattr(device_state, name)
        assert is_device_tensor(leaf), f"{name} must live on the device"
        assert leaf.device == device
    # Same seed, same program: the encoding graph is shared by both routes, so
    # generation 0 is bit-identical (only its location differs).
    for name in ("opcode", "payload", "size"):
        np.testing.assert_array_equal(
            to_host(getattr(device_state, name)), to_host(getattr(host_state, name))
        )
    np.testing.assert_array_equal(
        to_host(device_state.key), to_host(host_state.key)
    )
    assert np.all(np.isneginf(to_host(device_state.fitness)))


def test_init_population_puts_generation_zero_on_the_device(
    run_config, host_state, device
):
    """The encoding entry point itself (no driver in between) is route-neutral.

    ``init_gp`` is a thin forwarder, so this pins the layer the driver sits on:
    ``backend="iree"`` + an explicit device moves every leaf there and produces
    the same population as the numpy route for the same seed.
    """
    population = init_population(
        run_config, SEED, pop_size=POP_SIZE, backend="iree", device=device
    )
    for name in ("opcode", "payload", "size", "fitness", "key"):
        leaf = getattr(population, name)
        assert is_device_tensor(leaf), f"{name} must live on the device"
        assert leaf.device == device
    for name in ("opcode", "payload", "size"):
        np.testing.assert_array_equal(
            to_host(getattr(population, name)), to_host(getattr(host_state, name)),
            err_msg=f"{name} of generation 0 must be bit-identical across routes",
        )
    np.testing.assert_array_equal(to_host(population.key), to_host(host_state.key))
    assert np.all(np.isneginf(to_host(population.fitness)))


# --------------------------------------------------------------------------- #
# the operators: a device generation IS the numpy generation, bitwise
# --------------------------------------------------------------------------- #
def _tied_fitness(pop_size: int = POP_SIZE) -> np.ndarray:
    """``float32 [pop_size]`` MAXIMIZED fitness with ties and one ``-inf`` row.

    Rows 2/5/9/13 share the best value and are pairwise distinct trees in the
    fixture population, so "which row was kept" is decidable from the channels
    (see the tie test); rows 3/4/7/8/11/12 share the second value and row 14 is
    ``-inf`` (unevaluable).
    """
    fitness = np.array(
        [
            0.25, 0.25, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5,
            0.5, 1.0, 0.0, 0.5, 0.5, 1.0, -np.inf, 0.0,
        ],
        dtype=np.float32,
    )
    assert fitness.shape == (pop_size,), "the fixture vector is built for 16 rows"
    return fitness


#: The four spellings of ``variation`` the device route has to reproduce: pure
#: offspring, in-graph ranked elites, ranked elites fed a DEVICE fitness tensor
#: (no upload), and an explicit (deliberately non-ranked) ``elite_index``.
VARIATION_CASES = (
    pytest.param({"elite_count": 0}, id="offspring-only"),
    pytest.param({"elite_count": ELITE_COUNT}, id="ranked-elites"),
    pytest.param(
        {"elite_count": ELITE_COUNT, "device_fitness": True}, id="device-fitness"
    ),
    pytest.param({"elite_index": [13, 9]}, id="explicit-elite-index"),
)


@pytest.mark.parametrize("options", VARIATION_CASES)
def test_variation_on_a_device_state_matches_the_numpy_variation(
    run_config, host_state, device_state, device, options
):
    """Same key, same inputs, same offspring -- bitwise, on both channel kinds.

    The traced program is the same on both routes, and the operators move
    VALUES (gather/splice/gate): nothing on this path does float arithmetic, so
    the payload channel has to match bit for bit as well -- that is measured
    here rather than assumed, and an accidental float round trip anywhere in
    the device path would show up as a mismatch.
    """
    options = dict(options)
    device_fitness = options.pop("device_fitness", False)
    fitness = _tied_fitness()

    host_offspring, host_key = V.variation(
        run_config, host_state, fitness, SEED, **options
    )
    fitness_input = (
        evoxc.from_numpy(fitness).to(device) if device_fitness else fitness
    )
    device_offspring, device_key = V.variation(
        run_config,
        device_state,
        fitness_input,
        SEED,
        backend="iree",
        device=device,
        **options,
    )

    for name in ("opcode", "payload", "size"):
        np.testing.assert_array_equal(
            to_host(getattr(device_offspring, name)),
            to_host(getattr(host_offspring, name)),
            err_msg=f"{name} must be bit-identical across the two routes",
        )
    np.testing.assert_array_equal(
        to_host(device_offspring.fitness), to_host(host_offspring.fitness),
        err_msg="the next fitness vector is an in-graph output on both routes",
    )
    np.testing.assert_array_equal(to_host(device_key), to_host(host_key))
    np.testing.assert_array_equal(
        to_host(device_offspring.key), to_host(device_key)
    )
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert is_device_tensor(getattr(device_offspring, name))

    # Both routes agree on WHAT the case does, so the parity above is a parity
    # of the intended operator and not of two identical mistakes.
    recorded = to_host(device_offspring.fitness)
    if "elite_count" in options and options["elite_count"] == 0:
        assert np.all(np.isneginf(recorded)), "offspring-only: nothing evaluated"
    else:
        kept = (
            list(options["elite_index"])
            if "elite_index" in options
            else np.argsort(-fitness, kind="stable")[:ELITE_COUNT].tolist()
        )
        parent = {
            name: to_host(getattr(device_state, name))
            for name in ("opcode", "payload", "size")
        }
        child = {
            name: to_host(getattr(device_offspring, name))
            for name in ("opcode", "payload", "size")
        }
        for slot, source in enumerate(kept):
            for name in ("opcode", "payload", "size"):
                np.testing.assert_array_equal(
                    child[name][slot], parent[name][source],
                    err_msg=f"elite slot {slot} must be parent row {source}",
                )
        np.testing.assert_array_equal(recorded[: len(kept)], fitness[kept])
        assert np.all(np.isneginf(recorded[len(kept):]))


def test_the_in_graph_elite_ranking_reproduces_the_host_tie_rule(
    run_config, host_state, device_state, device
):
    """The subtle one: ties keep the LOWER row index, like the host rule.

    ``np.argsort(-fitness, kind="stable")`` breaks a tie towards the smaller
    row index, which the graph reproduces with ``argsort(-fitness,
    descending=False, stable=True)``.  A ``descending=True`` spell -- the
    obvious-looking one -- would hand back the tie group in REVERSE index order,
    so the fixture puts the best value on rows 2/5/9/13: the correct answer is
    ``[2, 5]`` and the flipped one ``[13, 9]``.  Those four rows are pairwise
    distinct trees, so the channels decide which rows were kept.
    """
    fitness = _tied_fitness()
    expected = np.argsort(-fitness, kind="stable")[:ELITE_COUNT]
    # The alternative rule this test must rule out: best-first, but a tie broken
    # towards the LAST row index (what `descending=True` does on the device).
    flipped = np.lexsort((-np.arange(fitness.size), -fitness))[:ELITE_COUNT]
    assert expected.tolist() == [2, 5], "fixture invariant: the host tie rule"
    assert flipped.tolist() == [13, 9], "the fixture can tell the two rules apart"

    names = ("opcode", "payload", "size")
    parent = {name: to_host(getattr(device_state, name)) for name in names}
    tied = [2, 5, 9, 13]
    for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        assert not np.array_equal(
            parent["opcode"][tied[left]], parent["opcode"][tied[right]]
        ) or not np.array_equal(
            parent["payload"][tied[left]], parent["payload"][tied[right]]
        ), "the tied rows must be distinguishable trees for this test to decide"

    device_offspring, _key = V.variation(
        run_config,
        device_state,
        fitness,
        SEED,
        elite_count=ELITE_COUNT,
        backend="iree",
        device=device,
    )
    child = {name: to_host(getattr(device_offspring, name)) for name in names}
    for slot, source in enumerate(expected):
        for name in names:
            np.testing.assert_array_equal(
                child[name][slot], parent[name][source],
                err_msg=(
                    f"elite slot {slot} must be parent row {source}: a tie keeps "
                    "the lower row index (a descending sort would pick "
                    f"{flipped[slot]})"
                ),
            )
    np.testing.assert_array_equal(
        to_host(device_offspring.fitness)[:ELITE_COUNT], fitness[expected]
    )

    # The numpy backend is the reference and must agree, row for row.
    host_offspring, _host_key = V.variation(
        run_config, host_state, fitness, SEED, elite_count=ELITE_COUNT
    )
    for name in names:
        np.testing.assert_array_equal(
            child[name], to_host(getattr(host_offspring, name))
        )


def test_a_nan_fitness_can_never_become_a_device_elite(
    run_config, device_state, device
):
    """The in-graph sanitation: a NaN ranks last, exactly like ``-inf``.

    A device generation has no host readback to clean the vector, so
    ``variation`` maps a non-finite fitness to ``-inf`` INSIDE the graph.  Row 2
    is the first elite of the fixture vector; with a NaN in it the elite must
    come from the ``stable`` ranking of the sanitized vector instead.
    """
    fitness = _tied_fitness()
    fitness[2] = np.nan
    sanitized = np.array(fitness, dtype=np.float32)
    sanitized[np.isnan(sanitized)] = -np.inf
    expected = np.argsort(-sanitized, kind="stable")[:ELITE_COUNT]
    assert 2 not in expected.tolist(), "fixture invariant: the NaN row loses"

    offspring, _key = V.variation(
        run_config,
        device_state,
        fitness,
        SEED,
        elite_count=ELITE_COUNT,
        backend="iree",
        device=device,
    )
    recorded = to_host(offspring.fitness)
    assert np.all(np.isfinite(recorded[:ELITE_COUNT])), (
        "a NaN must never reach the elite rows as a NaN"
    )
    assert not np.any(np.isnan(recorded))
    np.testing.assert_array_equal(recorded[:ELITE_COUNT], sanitized[expected])

    names = ("opcode", "payload", "size")
    parent = {name: to_host(getattr(device_state, name)) for name in names}
    child = {name: to_host(getattr(offspring, name)) for name in names}
    for slot, source in enumerate(expected):
        for name in names:
            np.testing.assert_array_equal(child[name][slot], parent[name][source])


# --------------------------------------------------------------------------- #
# the device objective: same numbers as the host problem
# --------------------------------------------------------------------------- #
def test_the_device_evaluator_matches_the_host_problem(
    tree_config, host_state, device_state, evaluator, data, device
):
    """``GraphEvaluator`` is the host ``SymbolicRegression``, on the device.

    The in-graph structure is traced from the channels and the body is shared
    with ``problem``'s in-graph route, so the two must agree -- measured at
    1.03e-07 relative on this box (iree contracts the odd multiply-add, so the
    device vector is not bitwise), while the numpy backend of the same class is
    bitwise exact against the host evaluator.
    """
    x, y = data
    problem = SymbolicRegression(x, y)
    host_channels = tuple(
        to_host(getattr(host_state, name)) for name in ("opcode", "payload", "size")
    )
    host_fitness = problem.evaluate(host_channels)

    raw = evaluator.evaluate(device_state)
    assert is_device_tensor(raw), "evaluate returns a DEVICE tensor (no sync)"
    assert tuple(int(dim) for dim in raw.shape) == (POP_SIZE,)
    device_fitness = graph_eval.readback_fitness(raw, POP_SIZE)
    assert device_fitness.dtype == np.float32
    np.testing.assert_allclose(
        device_fitness, host_fitness, rtol=1e-5, atol=0.0,
        err_msg="the device objective must reproduce the host problem's numbers",
    )

    # The readback is the documented one, and the class's other two spellings
    # agree with it (host_fitness = readback, evaluate_numpy = + readback).
    np.testing.assert_array_equal(evaluator.host_fitness(raw), device_fitness)
    np.testing.assert_array_equal(
        graph_eval.readback_fitness(raw), device_fitness
    )
    np.testing.assert_array_equal(
        evaluator.evaluate_numpy(device_state), device_fitness
    )

    # A population handed in as host rows (uploaded explicitly -- evoxc never
    # transfers implicitly) evaluates to the same numbers: the graph does not
    # care where its channels came from.
    uploaded = tuple(
        evoxc.from_numpy(to_host(getattr(device_state, name))).to(device)
        for name in ("opcode", "payload", "size")
    )
    np.testing.assert_array_equal(
        graph_eval.readback_fitness(evaluator.evaluate(uploaded), POP_SIZE),
        device_fitness,
    )

    # The numpy backend of the SAME class is the bitwise-exact oracle.  The
    # fixture population contains `/` nodes that divide by a zero-ish variable,
    # so the numpy kernel legitimately warns about the inf/`nan` it produces.
    numpy_evaluator = GraphEvaluator(x, y, used_ids=tree_config, backend="numpy")
    with np.errstate(all="ignore"):
        numpy_fitness = numpy_evaluator.evaluate_numpy(host_state)
    np.testing.assert_array_equal(numpy_fitness, host_fitness)
    assert evaluator.problem_dim == problem.problem_dim
    assert evaluator.solution_dim == problem.solution_dim


# --------------------------------------------------------------------------- #
# step: the same population as the numpy step, elites included
# --------------------------------------------------------------------------- #
def test_a_device_step_reproduces_the_numpy_step(
    run_config, host_state, device_state, evaluator, data, device
):
    x, y = data
    problem = SymbolicRegression(x, y)
    numpy_child = step(run_config, host_state, problem.evaluate)
    device_child = step(
        run_config, device_state, evaluator, backend="iree", device=device
    )

    # Both channel kinds are bitwise: the splice/selection/mutation graph is the
    # same program on both routes, and the payload channel is moved, never
    # computed (measured bitwise on the iree/rocm route).
    for name in ("opcode", "payload", "size"):
        np.testing.assert_array_equal(
            to_host(getattr(device_child, name)),
            to_host(getattr(numpy_child, name)),
            err_msg=f"{name} must be bit-identical across the two routes",
        )
    # And the device population is still a device population.
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert is_device_tensor(getattr(device_child, name))


def test_the_elite_rows_of_a_device_step_are_the_parents_own_top_rows(
    run_config, device_state, evaluator, device
):
    fitness = evaluator.host_fitness(evaluator.evaluate(device_state))
    device_child = step(
        run_config, device_state, evaluator, backend="iree", device=device
    )

    # `-fitness` + a stable argsort: the driver's own rule for the elite search.
    elite = np.argsort(-fitness, kind="stable")[:ELITE_COUNT]
    names = ("opcode", "payload", "size")
    parent = {name: to_host(getattr(device_state, name)) for name in names}
    child = {name: to_host(getattr(device_child, name)) for name in names}
    for slot, source in enumerate(elite):
        for name in names:
            np.testing.assert_array_equal(
                child[name][slot], parent[name][source],
                err_msg=f"elite slot {slot} must be a bit-identical parent row",
            )

    recorded = to_host(device_child.fitness)
    np.testing.assert_allclose(recorded[:ELITE_COUNT], fitness[elite], rtol=0, atol=TOL)
    assert np.all(np.isneginf(recorded[ELITE_COUNT:])), (
        "offspring are unevaluated until the next evaluate call"
    )


def test_a_host_objective_can_drive_a_device_population(
    run_config, device_state, data, device
):
    """The fitness KIND decides the transport, not the caller.

    A numpy-returning objective (the problem layer's host evaluator) on a
    device-resident population still works: the channels stay on the device, only
    the ``pop_size``-element fitness vector is uploaded, and the result is a
    device state again.
    """
    x, y = data
    device_child = step(
        run_config,
        device_state,
        SymbolicRegression(x, y).evaluate,
        backend="iree",
        device=device,
    )
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert is_device_tensor(getattr(device_child, name))
    recorded = to_host(device_child.fitness)
    assert np.all(np.isfinite(recorded[:ELITE_COUNT]))
    assert np.all(np.isneginf(recorded[ELITE_COUNT:]))


# --------------------------------------------------------------------------- #
# the budget: a device generation is sync-free, `run` is not
# --------------------------------------------------------------------------- #
def _instrument(monkeypatch):
    """Count host readbacks and channel downloads (an explicit transport probe).

    Both counters wrap the two functions every transfer in this stack goes
    through: :func:`evogp.gp.graph_eval.readback_fitness` (the D2H move of a
    fitness vector) and :func:`evogp.gp.encoding._as_numpy` (the D2H move of a
    channel).  Returns the two lists, which are filled as the wrapped calls run.
    """
    readbacks = []
    downloads = []
    real_readback = graph_eval.readback_fitness
    real_as_numpy = encoding._as_numpy

    def counting_readback(fitness, pop_size=None):
        readbacks.append(1)
        return real_readback(fitness, pop_size)

    def counting_as_numpy(value, name):
        if is_device_tensor(value):
            downloads.append(name)
        return real_as_numpy(value, name)

    monkeypatch.setattr(graph_eval, "readback_fitness", counting_readback)
    monkeypatch.setattr(encoding, "_as_numpy", counting_as_numpy)
    return readbacks, downloads


def test_a_device_step_takes_no_host_readback_and_no_channel_download(
    run_config, device_state, evaluator, device, monkeypatch
):
    """A device generation is TWO graph runs and nothing else.

    The elite rows are ranked and spliced INSIDE the variation graph and the
    next fitness vector is an in-graph output, so neither the fitness vector nor
    a channel has to cross to the host -- that is the whole optimization.  The
    instrumentation is proven live in the same test: the two wrapped functions
    really are the download path, so a ``0`` here is a real zero and not an
    un-instrumented code path.
    """
    readbacks, downloads = _instrument(monkeypatch)

    # The objective itself is sync-free too: `evaluate` returns a device tensor.
    raw = evaluator.evaluate(device_state)
    assert is_device_tensor(raw), "GraphEvaluator.evaluate must not sync"
    assert (readbacks, downloads) == ([], []), (
        "evaluating a population on the device must not touch the host "
        f"(readbacks={len(readbacks)}, downloads={downloads})"
    )

    child = step(run_config, device_state, evaluator, backend="iree", device=device)

    assert readbacks == [], (
        "a device generation has ZERO host readbacks -- the elite ranking and "
        "the next fitness vector are in-graph outputs (got "
        f"{len(readbacks)})"
    )
    assert downloads == [], (
        "no channel may leave the device inside a generation (the elite splice "
        f"runs in the variation graph); got downloads of {downloads}"
    )
    # The generation really happened: elites carry their fitness, offspring are
    # unevaluated, and the state is still a device state.
    recorded = to_host(child.fitness)
    assert np.all(np.isfinite(recorded[:ELITE_COUNT]))
    assert np.all(np.isneginf(recorded[ELITE_COUNT:]))
    assert is_device_tensor(child.opcode)

    # Positive control: the very same instrumentation sees the wrapped calls
    # (the assertions above already made a few of their own downloads while
    # reading the result, so restart the tally first).
    del readbacks[:]
    del downloads[:]
    graph_eval.readback_fitness(raw, POP_SIZE)
    encoding._as_numpy(child.opcode, "opcode")
    assert len(readbacks) == 1, "the readback counter must see a real readback"
    assert downloads == ["opcode"], "the download counter must see a real download"


def test_a_device_run_reads_back_one_vector_per_generation(
    run_config, evaluator, device, monkeypatch
):
    """The LOOP still reports: one fitness readback per evaluated generation.

    ``step`` is sync-free, but ``run`` builds the per-generation history and the
    best-tree report from the numbers, so it reads back exactly once per
    evaluated generation -- and touches the channels only twice, never inside
    the loop: the generation-0 validation and the final ``best_channels`` row.
    """
    generations = 2
    readbacks, downloads = _instrument(monkeypatch)

    run(
        run_config,
        evaluator,
        generations=generations,
        key=SEED,
        backend="iree",
        device=device,
    )

    assert len(readbacks) == generations, (
        "one fitness readback per evaluated generation, no more "
        f"(got {len(readbacks)} for {generations} generations)"
    )
    # Two channel round trips, neither of them inside the loop: `init_gp`'s
    # host-side validation of the generation-0 population, and the final
    # `GPResult.best_channels` report (one row download).
    assert downloads == ["opcode", "payload", "size"] * 2, downloads


def test_validate_false_drops_the_generation_zero_channel_download(
    run_config, evaluator, device, monkeypatch
):
    """``run(validate=False)`` starts sync-free and only downloads the report."""
    readbacks, downloads = _instrument(monkeypatch)

    run(
        run_config,
        evaluator,
        generations=1,
        key=SEED,
        backend="iree",
        device=device,
        validate=False,
    )

    assert len(readbacks) == 1
    assert downloads == ["opcode", "payload", "size"], downloads


# --------------------------------------------------------------------------- #
# run: the same trajectory, host rows in the result, state still on the device
# --------------------------------------------------------------------------- #
def test_a_device_run_reproduces_the_numpy_trajectory(
    run_config, evaluator, data, device
):
    generations = 3
    x, y = data
    device_result = run(
        run_config,
        evaluator,
        generations=generations,
        key=SEED,
        backend="iree",
        device=device,
    )
    host_result = run(
        run_config, SymbolicRegression(x, y).evaluate, generations=generations, key=SEED
    )

    assert device_result.generations == generations
    assert len(device_result.best_fitness_history) == generations
    np.testing.assert_allclose(
        np.asarray(device_result.best_fitness_history),
        np.asarray(host_result.best_fitness_history),
        rtol=0.0,
        atol=1e-5,
        err_msg="the two routes must follow the same trajectory",
    )
    assert np.all(np.diff(device_result.best_fitness_history) >= 0.0), (
        "elitism keeps the per-generation best monotone on the device too"
    )

    # The report is HOST data, whatever the population is made of...
    opcode, payload, size = device_result.best_channels
    for row, dtype in ((opcode, np.int32), (payload, np.float32), (size, np.int32)):
        assert isinstance(row, np.ndarray)
        assert row.dtype == dtype
        assert row.shape == (run_config[0].max_nodes,)
    # ...and re-evaluating it reproduces the reported fitness.
    replayed = SymbolicRegression(x, y).evaluate(
        (opcode[None, :], payload[None, :], size[None, :])
    )
    np.testing.assert_allclose(
        np.asarray(replayed, dtype=np.float32),
        np.float32(device_result.best_fitness),
        rtol=0.0,
        atol=1e-4,
    )
    # The state stays device-resident for whoever continues the run.
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert is_device_tensor(getattr(device_result.state, name))


# --------------------------------------------------------------------------- #
# the package + cache surface
# --------------------------------------------------------------------------- #
DEVICE_SURFACE = (
    "GraphEvaluator",
    "sr_fitness_channels",
    "readback_fitness",
    "FusedEvaluator",
    "fused_available",
)


def test_the_package_exposes_the_device_surface():
    import evogp.gp as gp

    missing = [name for name in DEVICE_SURFACE if name not in gp.__all__]
    assert not missing, f"evogp.gp does not export {missing}"
    for name in DEVICE_SURFACE:
        assert hasattr(gp, name)
    # The documented spellings still hold.
    assert gp.apply_variation is gp.variation.variation
    assert gp.apply_crossover is gp.crossover.crossover


def test_the_cache_surface_covers_the_device_layer(run_config, evaluator, device):
    clear_cache()
    run(
        run_config,
        evaluator,
        generations=1,
        key=SEED,
        backend="iree",
        device=device,
    )
    stats = cache_stats()
    assert "graph_eval" in stats["device_layers"]
    assert stats["device_layers"]["graph_eval"]["graphs"] > 0, (
        "a device run memoizes its evaluation graph"
    )
    assert stats["device_graphs"] == sum(
        layer["graphs"] for layer in stats["device_layers"].values()
    )
    assert set(stats["layers"]) == {
        "encoding",
        "selection",
        "crossover",
        "mutation",
        "variation",
        "workflow",
        "problem",
    }, "the host layer table keeps its seven entries"

    clear_cache()
    assert cache_stats()["device_graphs"] == 0
