"""The Evox 2.0 component contract of ``evogp.evox_ext``.

What is pinned here (the four properties the integration is defined by):

1. ``evox.core.StdWorkflow`` **accepts both configs and runs** -- it resolves
   the algorithm/problem functions from the module each config dataclass lives
   in, so a run is the proof the module convention is met;
2. the algorithm **state is tensors-only with constant shapes** -- every leaf is
   an ``evoxc`` tensor and the ``(shape, dtype)`` signature is identical in
   every generation (that is what lets one compiled step graph serve the run);
3. the **best-so-far fitness is monotone non-decreasing** over a short run and
   actually improves on the shipped target, and the in-graph fitness the
   workflow produced agrees with ``evogp.gp``'s own evaluator (elitism is
   carried inside the variation graph, so a step can never lose the incumbent);
4. the **compile happens once** -- ``StdWorkflow._step_exes`` holds exactly ONE
   entry after several generations, i.e. the step graph was traced/compiled for
   one state signature and then reused.

Everything runs on the numpy backend on the CPU, so the suite is hermetic and
needs no device. ``evox`` (and therefore ``evoxc``) is a hard requirement of the
integration: the whole module skips cleanly when ``evox`` is not importable.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("evox", reason="the Evox 2.0 integration needs evox")
evoxc = pytest.importorskip("evoxc", reason="evogp.evox_ext runs on evoxc")

import evox.core  # noqa: E402
from evox.core import StdWorkflow  # noqa: E402

from evogp.evox_ext import (  # noqa: E402
    OPT_DIRECTION,
    GPAlgorithm,
    GPAlgorithmConfig,
    SRProblemConfig,
    SRProblemState,
    accepts_cache_dir,
    best_so_far,
    generation_fitness,
    host_channels,
    make_gp_algorithm,
    make_sr_problem,
    make_stdworkflow,
)
from evogp.gp import graph_eval  # noqa: E402
from evogp.gp.config import GPConfig, GPTreeConfig  # noqa: E402
from evogp.gp.state import GPState  # noqa: E402

#: Data rows of the test problem (small: the numpy interpreter walks the
#: population node by node, so the cost tracks
#: ``pop_size * rows * max_nodes``).
N_ROWS = 128
#: Population size / tree width of the test configuration.
POP_SIZE = 32
MAX_NODES = 32
#: Elites carried per generation (``GPConfig.elite_count``).
ELITE_COUNT = 2
#: Generations the contract is checked over.
GENERATIONS = 4


def _data(seed: int = 0):
    """The shipped demo target ``x0**2 + x1**2`` on ``N_ROWS`` uniform points."""
    generator = np.random.default_rng(seed)
    x = generator.uniform(-5.0, 5.0, size=(N_ROWS, 2)).astype(np.float32)
    return x, (x[:, 0] ** 2 + x[:, 1] ** 2).astype(np.float32)


def _tree_config(max_nodes: int = MAX_NODES) -> GPTreeConfig:
    """The reference wiring of ``evogp.gp.smoke`` (torch-stack defaults)."""
    return GPTreeConfig(
        max_nodes=max_nodes,
        input_len=2,
        using_funcs=("+", "-", "*", "/"),
        max_layer_cnt=5,
        layer_leaf_prob=0.2,
        const_prob=0.5,
        const_range=(-5.0, 5.0),
        sample_cnt=8,
    )


def _gp_config(pop_size: int = POP_SIZE) -> GPConfig:
    return GPConfig(
        pop_size=pop_size,
        elite_count=ELITE_COUNT,
        tournament_size=4,
        crossover_prob=0.9,
        mutation_prob=0.2,
        mutation_kind="subtree",
    )


def _workflow(pop_size: int = POP_SIZE, max_nodes: int = MAX_NODES):
    """A workflow over the test problem, wired the way EvoGP means it."""
    x, y = _data()
    tree = _tree_config(max_nodes)
    algorithm = make_gp_algorithm(tree, _gp_config(pop_size))
    problem = make_sr_problem(x, y, used_ids=tree)
    return make_stdworkflow(algorithm, problem), (tree, problem, x, y)


def _signature(tree) -> tuple:
    """Ordered ``(shape, dtype)`` of every tensor leaf of a state pytree."""
    return tuple(
        (tuple(leaf.shape), str(leaf.dtype)) for leaf in evoxc.tree_leaves(tree)
    )


# --------------------------------------------------------------------------- #
# (a) the workflow accepts both configs and runs
# --------------------------------------------------------------------------- #
def test_workflow_accepts_both_configs_and_runs():
    workflow, (tree, problem, _x, _y) = _workflow()

    # The component contract: the workflow resolved each module from the config.
    assert workflow._algo_mod is __import__(
        "evogp.evox_ext.gp_algorithm", fromlist=["gp_algorithm"]
    )
    assert workflow._prob_mod is __import__(
        "evogp.evox_ext.sr_problem", fromlist=["sr_problem"]
    )

    state = workflow.init(seed=0)
    assert isinstance(state.algorithm_state, GPState)
    assert isinstance(state.problem_state, SRProblemState)
    assert int(np.asarray(state.generation.numpy())) == 0

    for expected in range(1, GENERATIONS + 1):
        state = workflow.step(state)
        assert int(np.asarray(state.generation.numpy())) == expected
        population = int(state.algorithm_state.opcode.shape[0])
        assert population == POP_SIZE
        assert int(state.algorithm_state.opcode.shape[1]) == int(tree.max_nodes)
        assert int(problem.n_rows) == N_ROWS

    # ... and the whole-loop entry point drives the same components (its
    # init_step/final_step fall back to `step`, which the module defines).
    looped = workflow.run(generations=2, seed=1)
    assert isinstance(looped.algorithm_state, GPState)
    assert len(workflow._step_exes) == 1


def test_configs_are_frozen_hashable_dataclasses():
    tree = _tree_config()
    algorithm = GPAlgorithm(tree=tree, gp=_gp_config())
    x, y = _data()
    problem = make_sr_problem(x, y, used_ids=tree)

    assert isinstance(algorithm, GPAlgorithmConfig)
    assert isinstance(problem, SRProblemConfig)
    for config in (algorithm, problem):
        assert config.__dataclass_params__.frozen
        assert isinstance(hash(config), int)

    # `make_*` constructors normalize array-like input into static tuples, so a
    # config is closed over by a traced graph as plain Python values.
    assert isinstance(problem.x, tuple) and isinstance(problem.y, tuple)
    assert isinstance(problem.used_ids, tuple) and len(problem.used_ids) == 4
    assert len(problem.x) == problem.n_rows * problem.input_len == N_ROWS * 2
    assert len(problem.y) == N_ROWS

    # The frozen pair refuses mutation (the graph is closed over these values).
    with pytest.raises(dataclasses.FrozenInstanceError):
        algorithm.gp = _gp_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.use_mse = False


def test_make_constructors_validate_their_arguments():
    tree = _tree_config()
    with pytest.raises(TypeError):
        make_gp_algorithm(tree)  # neither gp= nor pop_size=
    with pytest.raises(TypeError):
        make_gp_algorithm(tree, _gp_config(), pop_size=POP_SIZE)  # both
    with pytest.raises(ValueError):
        make_gp_algorithm(tree, pop_size=0)

    x, y = _data()
    with pytest.raises(ValueError):
        make_sr_problem(x, y[:-1])  # row mismatch
    with pytest.raises(ValueError):
        make_sr_problem(x, np.where(np.arange(N_ROWS) == 0, np.nan, y))
    with pytest.raises(ValueError):
        make_sr_problem(np.zeros((2, 2, 2)), [0.0, 1.0])


# --------------------------------------------------------------------------- #
# (b) tensors-only state, constant shapes
# --------------------------------------------------------------------------- #
def test_algorithm_state_leaves_are_evoxc_tensors_with_constant_shapes():
    workflow, _ = _workflow()
    state = workflow.init(seed=0)

    def assert_tensors(node):
        for leaf in evoxc.tree_leaves(node):
            assert isinstance(leaf, evoxc.Tensor), (
                f"state leaf {leaf!r} is not an evoxc tensor"
            )
        return _signature(node)

    initial = assert_tensors(state.algorithm_state)
    assert [shape for shape, _ in initial] == [
        (POP_SIZE, MAX_NODES),
        (POP_SIZE, MAX_NODES),
        (POP_SIZE, MAX_NODES),
        (POP_SIZE,),
        (),
    ]
    assert [dtype for _, dtype in initial] == [
        "int32",
        "float32",
        "int32",
        "float32",
        "int64",
    ]

    for _ in range(GENERATIONS):
        state = workflow.step(state)
        assert assert_tensors(state.algorithm_state) == initial

    # The problem state is tensors-only too: the data rides as runtime inputs
    # of the step graph (uploaded once by `init`), never as graph constants.
    problem_signature = assert_tensors(state.problem_state)
    assert [shape for shape, _ in problem_signature] == [(N_ROWS, 2), (N_ROWS,)]
    assert [dtype for _, dtype in problem_signature] == ["float32", "float32"]


# --------------------------------------------------------------------------- #
# (c) monotone best-so-far, real improvement, and truth parity
# --------------------------------------------------------------------------- #
def test_best_so_far_is_monotone_and_improves():
    workflow, _ = _workflow()
    state = workflow.init(seed=0)

    history = []
    for _ in range(GENERATIONS):
        state = workflow.step(state)
        history.append(best_so_far(state))

    assert all(np.isfinite(history)), history
    assert all(later >= earlier for earlier, later in zip(history, history[1:])), history
    # The shipped target improves well inside 4 generations at this size.
    assert history[-1] > history[0], history

    # The step graph wrote exactly `elite_count` known values; every offspring
    # row is the `-inf` "not evaluated yet" marker.
    fitness = generation_fitness(state)
    assert fitness.shape == (POP_SIZE,)
    assert int(np.isfinite(fitness).sum()) == ELITE_COUNT


def test_in_graph_fitness_matches_the_gp_evaluator():
    workflow, (tree, _problem, x, y) = _workflow()
    state = workflow.init(seed=0)

    def truth(workflow_state):
        """The evalued best fitness, through evogp.gp's public in-graph route."""
        return float(
            graph_eval.sr_fitness_channels_numpy(
                workflow_state.algorithm_state, x, y, used_ids=tree
            ).max()
        )

    # Elitism carries the incumbent bit-identically, so the TRUE best fitness
    # can never fall -- this is what makes the workflow's own vector trustworthy.
    history = [truth(state)]
    for generation in range(GENERATIONS):
        state = workflow.step(state)
        history.append(truth(state))
        # The step graph wrote the elite values of the population it evaluated,
        # i.e. exactly the best fitness of the generation before it.
        assert best_so_far(state) == pytest.approx(history[generation], rel=1e-6)

    assert all(later >= earlier for earlier, later in zip(history, history[1:])), history
    assert history[-1] > history[0], history


def test_host_channels_matches_the_workflow_state():
    workflow, _ = _workflow()
    state = workflow.step(workflow.init(seed=0))

    opcode, payload, size = host_channels(state)
    assert opcode.shape == (POP_SIZE, MAX_NODES)
    assert opcode.dtype == np.int32
    assert payload.dtype == np.float32
    assert size.dtype == np.int32
    # Every row's live node count is its root's subtree size, in (0, max_nodes].
    lengths = size[:, 0]
    assert np.all(lengths >= 1) and np.all(lengths <= MAX_NODES)


# --------------------------------------------------------------------------- #
# (d) one compile for the whole run
# --------------------------------------------------------------------------- #
def test_step_graph_is_compiled_once():
    workflow, _ = _workflow()
    state = workflow.init(seed=0)
    assert len(workflow._step_exes) == 1  # built by init()

    for _ in range(3):
        state = workflow.step(state)

    assert len(workflow._step_exes) == 1, list(workflow._step_exes)
    ((variant, signature), _executable) = next(iter(workflow._step_exes.items()))
    assert variant == "step"
    # 5 algorithm leaves (three channels + fitness + key), 2 problem leaves (the
    # data and the targets) and the workflow's own 3 (monitor placeholder,
    # generation counter, key).
    assert len(signature) == 10


# --------------------------------------------------------------------------- #
# the orientation the pair depends on
# --------------------------------------------------------------------------- #
def test_make_stdworkflow_enforces_the_maximization_orientation():
    x, y = _data()
    algorithm = GPAlgorithm(tree=_tree_config(), gp=_gp_config())
    problem = make_sr_problem(x, y)
    assert OPT_DIRECTION == "max"

    with pytest.raises(ValueError, match="opt_direction"):
        make_stdworkflow(algorithm, problem, opt_direction="min")

    # A hand-built workflow can still be mis-wired (nothing can stop that), but
    # the components' own state contract is unchanged by it.
    manual = StdWorkflow(algorithm, problem, opt_direction="min")
    state = manual.step(manual.init(seed=0))
    assert isinstance(state.algorithm_state, GPState)
    assert _signature(state.algorithm_state) == _signature(
        make_stdworkflow(algorithm, problem).init(seed=0).algorithm_state
    )


# --------------------------------------------------------------------------- #
# the step-artifact cache (evox's opt-in `StdWorkflow(cache_dir=...)`)
# --------------------------------------------------------------------------- #
class _RecordingWorkflow:
    """A stand-in for `evox.core.StdWorkflow` that records what it was handed.

    The two subclasses below are the two worlds the adapter has to serve: an
    `evox` that carries the opt-in step-artifact cache (`cache_dir=` among its
    keywords) and one that predates it (no such keyword at all, so an
    unexpected argument would raise ``TypeError``).
    """

    def __init__(self, algorithm, problem, **kwargs):
        self.algorithm = algorithm
        self.problem = problem
        self.kwargs = dict(kwargs)
        self.cache_dir = kwargs.get("cache_dir")


class _RecordingWorkflowWithCache(_RecordingWorkflow):
    """The workflow of an `evox` that declares ``cache_dir``."""

    def __init__(
        self,
        algorithm,
        problem,
        monitor=None,
        opt_direction="min",
        num_generations=None,
        backend="numpy",
        device=None,
        compile_options=None,
        cache_dir=None,
    ):
        super().__init__(
            algorithm,
            problem,
            monitor=monitor,
            opt_direction=opt_direction,
            num_generations=num_generations,
            backend=backend,
            device=device,
            compile_options=compile_options,
            cache_dir=cache_dir,
        )


class _RecordingWorkflowWithoutCache(_RecordingWorkflow):
    """The workflow of an `evox` that declares no ``cache_dir``."""

    def __init__(
        self,
        algorithm,
        problem,
        monitor=None,
        opt_direction="min",
        num_generations=None,
        backend="numpy",
        device=None,
        compile_options=None,
    ):
        super().__init__(
            algorithm,
            problem,
            monitor=monitor,
            opt_direction=opt_direction,
            num_generations=num_generations,
            backend=backend,
            device=device,
            compile_options=compile_options,
        )


def _pair():
    """The two configs of the test problem, without building a workflow."""
    tree = _tree_config()
    algorithm = make_gp_algorithm(tree, _gp_config())
    return algorithm, make_sr_problem(*_data(), used_ids=tree)


def test_accepts_cache_dir_is_a_signature_probe():
    """The forward is feature-detected, never pinned to one `evox` release."""
    assert accepts_cache_dir(_RecordingWorkflowWithCache) is True
    assert accepts_cache_dir(_RecordingWorkflowWithoutCache) is False
    # Nothing introspectable: answer "do not pass it" instead of raising.
    assert accepts_cache_dir(object) is False


def test_make_stdworkflow_forwards_cache_dir_when_evox_declares_it(monkeypatch, tmp_path):
    """The opt-in step-artifact cache is reachable through the adapter."""
    monkeypatch.setattr(evox.core, "StdWorkflow", _RecordingWorkflowWithCache)
    algorithm, problem = _pair()
    workflow = make_stdworkflow(
        algorithm,
        problem,
        backend="iree",
        device="rocm:0",
        num_generations=7,
        cache_dir=tmp_path,
    )

    # The cache directory arrives verbatim, and nothing else is lost on the way.
    assert workflow.cache_dir == tmp_path
    assert workflow.kwargs == {
        "monitor": None,
        "opt_direction": OPT_DIRECTION,
        "num_generations": 7,
        "backend": "iree",
        "device": "rocm:0",
        "compile_options": None,
        "cache_dir": tmp_path,
    }
    # `None` is the historical build-every-time default, forwarded as well.
    assert make_stdworkflow(algorithm, problem).cache_dir is None


def test_make_stdworkflow_without_the_cache_dir_kwarg_degrades(monkeypatch, tmp_path):
    """An `evox` that declares no `cache_dir` still builds a workflow."""
    monkeypatch.setattr(evox.core, "StdWorkflow", _RecordingWorkflowWithoutCache)
    algorithm, problem = _pair()

    with pytest.warns(RuntimeWarning, match="cache_dir"):
        workflow = make_stdworkflow(algorithm, problem, cache_dir=tmp_path)
    # Degraded, not raised: a workflow comes back without the keyword, and
    # everything else was forwarded unchanged.
    assert "cache_dir" not in workflow.kwargs
    assert workflow.kwargs["opt_direction"] == OPT_DIRECTION
    assert workflow.kwargs["backend"] == "numpy"

    # Without a request there is nothing to drop, so nothing is warned about.
    workflow = make_stdworkflow(algorithm, problem)
    assert "cache_dir" not in workflow.kwargs


def test_the_installed_evox_decides_whether_the_cache_dir_is_forwarded(tmp_path):
    """Against the real `evox`: forwarded when declared, degraded (not raised) else."""
    algorithm, problem = _pair()
    if accepts_cache_dir(StdWorkflow):
        workflow = make_stdworkflow(algorithm, problem, cache_dir=str(tmp_path))
        assert workflow.cache_dir == str(tmp_path)
    else:
        with pytest.warns(RuntimeWarning, match="cache_dir"):
            workflow = make_stdworkflow(algorithm, problem, cache_dir=str(tmp_path))
        assert not hasattr(workflow, "cache_dir")
    # Either way the workflow is a real one over the pair.
    assert workflow.algorithm is algorithm and workflow.problem is problem
