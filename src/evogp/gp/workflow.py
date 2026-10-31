"""The driver of the evoxc GP stack: ``init_gp`` / ``step`` / ``run``.

This is the last layer of the functional stack: the encoding
(:mod:`evogp.gp.encoding`), the operator layer (:mod:`evogp.gp.variation`) and
the problem layer (:mod:`evogp.gp.problem`) are plugged together here by the
step protocol of the Evox 2.0 idiom (``DESIGN.md`` section "Evaluation and the
driver")::

    state = init_gp(config, key)                       # one generation-0 population
    state = step(config, state, evaluate)              # one generation
    result = run(config, evaluate, generations=100)    # the loop

``evaluate`` is the whole plug-in surface: any callable
``channels -> (pop_size,) float32`` (MAXIMIZED, losses negated, non-finite =
``-inf``).  A :class:`evogp.gp.problem.SymbolicRegression` /
:class:`evogp.gp.problem.CustomLoss` instance satisfies it twice over -- the
object is itself callable (``__call__`` is ``evaluate``) and it exposes
``.evaluate`` (the numpy host evaluator) plus ``.evaluate_graph`` (the same
fitness from one traced ``evoxc`` graph) -- and the torch engine satisfies it
through the caller-side adapter (``to_legacy_triple`` -> ``../tree/Forest`` ->
``Forest.SR_fitness``), which lives outside this subtree because it needs torch.

Elitism is the driver's job
---------------------------
:func:`evogp.gp.variation.variation` returns **offspring only** -- it selects
parents, crosses and mutates.  Exactly like
``../algorithm/genetic_programming.py::GeneticProgramming.step``, the best
``GPConfig.elite_count`` individuals of the evaluated generation are copied into
the next generation *unchanged* (bit-identical rows, read straight out of the
parent state), and the offspring fill the remaining rows.  Elitism is what makes
the per-generation best fitness monotone non-decreasing, which is the property
:class:`GPResult` reports.

The operator work stays in graphs, the evaluation stays pluggable
-----------------------------------------------------------------
``step`` runs the operator layer's traced graph -- ONE ``evoxc.run``, because
that graph also splices the elite rows in front of the offspring and writes the
next fitness vector.  There is no host round trip inside the operator step and no
traced ``evaluate``.  The evaluation is a plain Python call, so the driver's own
host-readback budget is **one fitness vector per evaluated generation** when the
numbers are actually needed (the history :func:`run` builds); :func:`step` on a
device population needs none at all.

Device-resident runs
--------------------
A run whose population lives as ``evoxc`` tensors ON A DEVICE stays there for
the whole loop, and the driver is what makes that possible end to end:

* :func:`init_gp` returns the device state directly
  (``encoding.init_population(..., backend="iree", device="rocm")``);
* the objective is the device-resident in-graph evaluator
  (:class:`evogp.gp.graph_eval.GraphEvaluator`), which returns its
  ``(pop_size,)`` fitness as a DEVICE tensor and never syncs -- so the useful
  work of a generation (variation + evaluation) is two ``evoxc.run`` calls;
* :func:`step` hands that device vector straight to the variation graph and
  never reads it back: the graph ranks the elite rows itself (``argsort`` of
  ``-fitness``, stable, i.e. the host tie rule) and returns the next fitness
  vector, so a device ``step`` is **two graph runs, zero host syncs, zero
  host<->device transfers**;
* :func:`run` is the one place that reads the fitness back -- once per evaluated
  generation, for the per-generation best/median history -- and it downloads the
  channels exactly once more, at the end, to report
  :attr:`GPResult.best_channels` as host rows.

Both fitness kinds are accepted from the SAME ``evaluate`` callable: a numpy
array (the host path, unchanged) or an ``evoxc`` tensor, whose device is then
the run's device.  Nothing has to be declared: an objective that returns a
device tensor drives the device path, and the fitness contract (MAXIMIZED,
non-finite = ``-inf``) is applied either on the readback or in-graph
(``variation`` maps NaN to ``-inf`` before it ranks or selects).

The RNG is the state
--------------------
``GPState.key`` is the *next* generation's key: ``init_population`` produces the
generation-0 key, :func:`variation` advances it in-graph and returns it, and the
driver stores it in the next state.  A run is therefore reproducible from its
seed alone, and two runs with the same seed are bit-identical.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Tuple

import numpy as np

from . import _runtime, encoding, graph_eval
from . import variation as _variation
from .config import GPConfig, GPTreeConfig
from .state import UNEVALUATED, GPState, population_size

__all__ = [
    "GPResult",
    "init_gp",
    "step",
    "run",
    "cache_stats",
    "clear_cache",
]

#: The driver's own graph memo.  It is empty by construction: the elite splice
#: lives in the variation graph (``variation(..., elite_count=...)``), so the
#: driver traces nothing.  The dict stays so ``cache_stats()`` keeps reporting
#: its ``workflow`` layer, and the aggregate surface keeps its seven entries.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the driver layer (no graph of its own since the elite fusion).

    ``graphs`` is always ``0``: the driver's only traced executable was the
    elite splice, which :func:`evogp.gp.variation.variation` now performs inside
    the generation graph.  The key keeps the aggregate cache surface stable.
    """
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop the driver's memo (empty; the operator graphs are their own memos)."""
    _runtime.reset(_EXECUTABLE_MEMO)


# --------------------------------------------------------------------------- #
# the result object
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class GPResult:
    """What a :func:`run` returns.

    Args:
        best_fitness: Highest fitness ever evaluated -- ``max(best_fitness_history)``
            (``-inf`` when no generation produced a finite fitness).
        best_fitness_history: One entry per evaluated generation: the best
            fitness of *that* generation.  Monotone non-decreasing because of
            elitism (the elite of generation ``g`` is a bit-identical row of
            generation ``g + 1``).
        median_fitness_history: One entry per evaluated generation: the median
            over the *finite* fitness values of that generation (``-inf`` for a
            generation in which every individual was non-finite).  The median
            rather than the mean because the fitness of a GP population is
            heavy-tailed -- a few exploding trees drag a mean over many orders
            of magnitude, so it says nothing about the population.
        best_channels: ``(opcode, payload, size)`` numpy arrays of the row that
            achieved :attr:`best_fitness` (``None`` when nothing was ever
            finite).  Each array is a copy of shape ``[max_nodes]``, i.e. the
            three 1-D channel rows of ONE individual; pass the tuple back
            unchanged to re-evaluate that tree (``problem.evaluate`` /
            ``predict`` and :func:`evogp.gp.encoding.validate_channels` /
            :func:`evogp.gp.encoding.to_legacy_triple` read it as a population
            of one).  They are HOST rows even for a device-resident run: the
            driver downloads the winning row once, at the end.
        best_generation: 0-based index of the generation that produced
            :attr:`best_channels`.
        generations: Number of generation steps actually performed.
        state: The final population state (the generation after the last
            evaluated one; ``run(generations=0)`` returns the initial
            population).  A device run returns device-resident channels.
    """

    best_fitness: float
    best_fitness_history: Tuple[float, ...]
    median_fitness_history: Tuple[float, ...]
    best_channels: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    best_generation: int
    generations: int
    state: Optional[GPState]


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #
def init_gp(
    config,
    key,
    *,
    pop_size: Optional[int] = None,
    backend: str = "numpy",
    device: str = "cpu",
    validate: bool = True,
) -> GPState:
    """The generation-0 population of a run (:func:`evogp.gp.init_population`).

    Args:
        config: a ``(GPTreeConfig, GPConfig)`` pair (the normal case -- then
            ``pop_size`` comes from the ``GPConfig``) or a bare
            :class:`~evogp.gp.config.GPTreeConfig` plus ``pop_size``.
        key: an int seed or a concrete ``evoxc`` key tensor.  The returned
            state's ``key`` is the *next* generation's key.
        pop_size: number of trees; see ``config``.
        backend: evoxc backend name (``"numpy"`` default).
        device: evoxc device spelling (``"cpu"`` default, ``"rocm"`` /
            ``"rocm:0"`` on this AMD box) or an ``evoxc.Device``.
        validate: run the host-side encoding validation on the result.

    Returns:
        A :class:`~evogp.gp.state.GPState` with an all-``-inf`` fitness vector.
        ``backend``/``device`` naming a real device (e.g. ``backend="iree",
        device="rocm"``) makes that state DEVICE-RESIDENT -- its channels, its
        fitness vector and its key all live on the device, which is what a
        device-resident run (:func:`step` / :func:`run` with a
        :class:`evogp.gp.graph_eval.GraphEvaluator`) is built from.
    """
    return encoding.init_population(
        config,
        key,
        pop_size=pop_size,
        backend=backend,
        device=device,
        validate=validate,
    )


# --------------------------------------------------------------------------- #
# one generation
# --------------------------------------------------------------------------- #
def _configs(config, pop_size: Optional[int]) -> Tuple[GPTreeConfig, GPConfig]:
    """``(GPTreeConfig, GPConfig)`` of a run, validated against ``pop_size``."""
    tree_config, gp_config, resolved_pop = _runtime.resolve_configs(
        config, pop_size=pop_size
    )
    if tree_config is None:
        raise ValueError(
            "the driver needs a GPTreeConfig (or a (GPTreeConfig, GPConfig) "
            "pair): the variation operator draws from the function set, the "
            "constant table and input_len"
        )
    if gp_config is None:
        if resolved_pop is None:
            raise ValueError(
                "pop_size is required with a bare GPTreeConfig (pass a "
                "(GPTreeConfig, GPConfig) pair to take it from the GPConfig)"
            )
        gp_config = GPConfig(pop_size=int(resolved_pop))
    return tree_config, gp_config


def _fitness(state: GPState, evaluate: Callable, read_back: bool = True):
    """The host readback of one evaluation (the driver's only sync point).

    Args:
        state: the population to evaluate.
        evaluate: the objective.
        read_back: whether the fitness numbers are needed on the host.  A device
            generation that lets the operator layer rank the elite rows itself
            does not need them (:func:`step` passes ``read_back=False``), so it
            skips the device-to-host move entirely; ``run`` always reads back,
            because the history is built from the numbers.

    Returns:
        ``(values, raw)`` -- the ``[pop_size]`` float32 host fitness (fresh,
        writable, non-finite mapped to ``-inf``; ``None`` when
        ``read_back=False``) and the object ``evaluate`` returned, which
        :func:`_variation_fitness` hands back to the operator layer unchanged
        when it already lives on the device.

    The two accepted kinds are told apart by :func:`evogp.gp._runtime.is_tensor`,
    never by asking the caller: a numpy array takes the host path the driver has
    always had, an ``evoxc`` tensor is read back through
    :func:`evogp.gp.graph_eval.readback_fitness` (one sync, and a device buffer
    is moved to the host first because ``evoxc`` never transfers implicitly).
    """
    if not callable(evaluate):
        raise TypeError(
            f"evaluate must be callable (channels -> [pop_size] fitness), got "
            f"{type(evaluate).__name__}"
        )
    expected = population_size(state)
    raw = evaluate(state)
    if _runtime.is_tensor(raw):
        shape = tuple(int(dim) for dim in raw.shape)
        if len(shape) != 1 or shape[0] != expected:
            raise ValueError(
                f"evaluate returned shape {shape}; the contract is a "
                f"[pop_size={expected}] float32 fitness vector"
            )
        if not read_back:
            return None, raw
        # The ONE readback of a generation. `readback_fitness` owns the D2H move
        # and the repo-wide non-finite -> -inf rule (`StandardPipeline.step` maps
        # NaN to -inf in place; the driver does the same so an unevaluable tree
        # can never win a tournament).
        return graph_eval.readback_fitness(raw, expected), raw
    values = np.array(
        encoding._as_numpy(raw, "fitness"), dtype=np.float32, copy=True
    )
    if values.ndim != 1 or int(values.shape[0]) != expected:
        raise ValueError(
            f"evaluate returned shape {values.shape}; the contract is a "
            f"[pop_size={expected}] float32 fitness vector"
        )
    values[np.isnan(values)] = float(UNEVALUATED)
    return values, raw


def _variation_fitness(raw, values):
    """The fitness vector handed to :func:`evogp.gp.variation.variation`.

    A device objective just produced a device tensor, so passing that tensor on
    saves the ``pop_size``-element upload a device generation would otherwise
    pay; the operator layer maps non-finite values to ``-inf`` IN-GRAPH
    (:func:`evogp.gp.variation._sanitize_fitness`), so the device vector needs no
    host-side cleaning first.  ``values is None`` is the sync-free step path,
    which has nothing but the tensor.  Anything that is not a ``float32``
    ``evoxc`` tensor falls back to the host vector, which is always valid input.
    """
    if values is None:
        return raw
    if not _runtime.is_tensor(raw):
        return values
    if np.dtype(raw.dtype) != np.dtype(np.float32):
        return values
    return raw


def _state_device(state: GPState):
    """The device a state's channels live on, or ``None`` for a host population.

    Reads the tensors' own metadata only (``Tensor.device`` is a host-side field,
    not a sync), which is how the driver decides between the two routes: a
    device-resident population never downloads its channels, and a host
    population keeps the byte-for-byte path it has always had.
    """
    channels = _runtime.as_channels(state)
    if not all(_runtime.is_tensor(channel) for channel in channels):
        return None
    device = channels[0].device
    return None if device.kind == "cpu" else device


def _advance(config, state: GPState, fitness, key=None,
             backend: str = "numpy", device: str = "cpu",
             fitness_input=None) -> GPState:
    """The next generation from an already evaluated ``state``.

    One call of the operator layer, nothing else: :func:`evogp.gp.variation.
    variation` selects the parents, crosses, mutates, and -- with the config's
    ``elite_count`` -- prepends the elite rows and writes the next fitness
    vector in the SAME graph, so the next state is complete when it returns
    (channels, fitness, advanced key) with no splice run and no upload in
    between.

    Args:
        config: the run's config.
        state: the evaluated population.
        fitness: its host fitness readback (``[pop_size]``, sanitized), or
            ``None`` on the sync-free path, which passes no host numbers at all.
        key: the RNG key to vary with (``None`` continues ``state.key``).
        backend: evoxc backend name.
        device: evoxc device spelling.
        fitness_input: the object ``evaluate`` returned, when it is an ``evoxc``
            tensor -- :func:`_variation_fitness` decides whether it replaces the
            host vector (no upload on a device generation).
    """
    _tree_config, gp_config = _configs(config, population_size(state))
    offspring, _next_key = _variation.variation(
        config,
        state,
        _variation_fitness(fitness_input, fitness),
        state.key if key is None else key,
        elite_count=int(gp_config.elite_count),
        backend=backend,
        device=device,
    )
    return offspring


def step(
    config,
    state: GPState,
    evaluate: Callable,
    *,
    key=None,
    backend: str = "numpy",
    device: str = "cpu",
) -> GPState:
    """One generation: evaluate ``state``, then select / cross / mutate.

    Mirrors ``GeneticProgramming.step``: evaluate the current population, keep
    the best ``GPConfig.elite_count`` individuals verbatim, and let
    :func:`evogp.gp.variation.variation` fill the rest with offspring.  The
    elite is spliced inside the *same* graph that assembles the next population,
    so the elite rows are bit-identical copies and a generation is ONE operator
    run -- no second executable for the splice.

    Args:
        config: a ``(GPTreeConfig, GPConfig)`` pair (or a bare
            ``GPTreeConfig``, then the ``GPConfig`` defaults apply).
        state: the population to evaluate (shapes must match the config).
        evaluate: ``channels -> (pop_size,) float32``, MAXIMIZED, non-finite =
            ``-inf``.  Called exactly once per ``step`` and never traced.  A
            ``SymbolicRegression`` / ``CustomLoss`` instance, its ``.evaluate``,
            its ``.evaluate_graph`` or any engine adapter all qualify.  The
            result may be a numpy array (the host path) or an ``evoxc`` tensor;
            a DEVICE tensor is what makes the whole generation device-resident --
            nothing has to be declared, the driver detects the kind
            (:class:`evogp.gp.graph_eval.GraphEvaluator` is the device
            objective).
        key: the RNG key to vary with; ``None`` (the default) continues the
            state's own key, which is what a loop wants.
        backend: evoxc backend name (``"numpy"`` default).
        device: evoxc device spelling (``"cpu"`` default).

    Returns:
        The next generation: channels of ``elite_count`` elite rows followed by
        the offspring rows, a fitness vector that carries the elite's known
        fitness and ``-inf`` for every offspring, and the advanced key.  For a
        device-resident ``state`` all of these live on the same device and
        NEITHER the channels nor the fitness vector cross to the host: the
        variation graph ranks the elite rows itself (``-fitness``, stable, so the
        tie rule is the host rule), which is why a device ``step`` has **zero**
        host syncs and zero host transfers -- two graph runs (`evaluate` + this
        one) and nothing else.  A host population keeps the documented host
        path, where the readback is a plain numpy array and costs nothing.

    Raises:
        ValueError: when ``evaluate`` returns the wrong shape and when the state
            disagrees with the config; ``TypeError`` when ``evaluate`` is not
            callable.
    """
    # A device-resident population needs no host numbers: `evaluate` hands back
    # a device tensor, the variation graph turns it into the elite ranking, and
    # the next fitness vector is an in-graph output.  So skip the readback
    # (`read_back=False`) -- that is the one sync of a generation, gone.
    read_back = _state_device(state) is None
    values, raw = _fitness(state, evaluate, read_back=read_back)
    return _advance(
        config,
        state,
        values,
        key=key,
        backend=backend,
        device=device,
        fitness_input=raw,
    )


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def _row_numpy(state: GPState, row: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One individual of ``state`` as fresh ``int32``/``float32``/``int32`` rows.

    The one download of a device run (``encoding._as_numpy`` moves a device
    buffer to the host explicitly), and it happens once, after the loop, when
    :class:`GPResult` is assembled.
    """
    opcode, payload, size = state.channels()
    return (
        np.array(encoding._as_numpy(opcode, "opcode"), dtype=np.int32)[row].copy(),
        np.array(encoding._as_numpy(payload, "payload"), dtype=np.float32)[row].copy(),
        np.array(encoding._as_numpy(size, "size"), dtype=np.int32)[row].copy(),
    )


def run(
    config,
    evaluate: Callable,
    generations: int = 100,
    *,
    key=0,
    pop_size: Optional[int] = None,
    backend: str = "numpy",
    device: str = "cpu",
    callback: Optional[Callable] = None,
    validate: bool = True,
) -> GPResult:
    """Run ``generations`` generations of the driver and report the best tree.

    ``generations`` counts *evaluated* generations: the loop starts from a fresh
    population (:func:`init_gp`), evaluates it, advances it, and repeats, so
    ``generations=0`` returns the initial population unevaluated and
    ``generations=1`` evaluates exactly one population.

    Args:
        config: a ``(GPTreeConfig, GPConfig)`` pair (or a bare
            ``GPTreeConfig`` plus ``pop_size``).
        evaluate: ``channels -> (pop_size,) float32`` (MAXIMIZED); see
            :func:`step`.  A numpy-returning objective runs the host loop, a
            device-tensor-returning one (``GraphEvaluator``) runs the
            device-resident loop; the two are interchangeable per call.
        generations: number of generations to run (``>= 0``).
        key: the int seed (or concrete ``evoxc`` key) of generation 0.  The rest
            of the RNG stream is carried in the state, so this is the only knob
            a reproducible run needs.
        pop_size: only for a bare ``GPTreeConfig`` (a pair carries it).
        backend: evoxc backend name (``"numpy"`` default).
        device: evoxc device spelling (``"cpu"`` default, ``"rocm"`` /
            ``"rocm:0"`` on this AMD box) or an ``evoxc.Device``.
        callback: optional ``callback(generation, state, fitness)`` called after
            every evaluated generation with the 0-based generation index, the
            population that was just evaluated and its ``[pop_size]`` float32
            numpy fitness readback (the same array the history is built from).
        validate: run ``init_gp``'s host-side encoding validation on the
            generation-0 population (default ``True``).  It is the only channel
            download before the loop and costs one ``pop_size x max_nodes`` round
            trip; a run that has already validated its configuration can turn it
            off with ``validate=False`` and start fully sync-free.

    Returns:
        A :class:`GPResult`.  ``result.best_fitness_history`` is the
        per-generation best (monotone, see the class docstring),
        ``result.best_channels`` the winning tree as HOST numpy rows (downloaded
        once, at the end, even for a device run), and ``result.state`` the final
        population -- device-resident when ``backend``/``device`` (or a device
        ``evaluate``) made the run a device run.

    Raises:
        ValueError: for a negative / non-int ``generations``, a mismatched
            config, or an ``evaluate`` that breaks the fitness contract.

    Notes:
        Host syncs: ONE fitness readback per evaluated generation (the history
        is built from it -- that is the documented reporting contract), plus the
        final :attr:`GPResult.best_channels` download; ``validate=False`` drops
        the generation-0 validation download.  A device generation costs two
        graph runs (``evaluate`` + ``variation``), no splice run.
    """
    if not isinstance(generations, int) or isinstance(generations, bool):
        raise ValueError(f"generations must be an int, got {generations!r}")
    if generations < 0:
        raise ValueError(f"generations must be >= 0, got {generations}")
    _configs(config, pop_size)

    state = init_gp(
        config, key, pop_size=pop_size, backend=backend, device=device,
        validate=validate,
    )

    best_fitness = float(UNEVALUATED)
    best_history: list = []
    median_history: list = []
    best_row = -1
    best_generation = -1
    best_state: Optional[GPState] = None

    for generation in range(generations):
        evaluated = state
        fitness, raw = _fitness(evaluated, evaluate)

        generation_best = float(np.max(fitness))
        best_history.append(generation_best)
        finite = fitness[np.isfinite(fitness)]
        median_history.append(
            float(np.median(finite)) if finite.size else float(UNEVALUATED)
        )
        if generation_best > best_fitness:
            best_fitness = generation_best
            best_row = int(np.argmax(fitness))
            best_generation = generation
            best_state = evaluated
        if callback is not None:
            callback(generation, evaluated, fitness)

        state = _advance(
            config,
            evaluated,
            fitness,
            backend=backend,
            device=device,
            fitness_input=raw,
        )

    best_channels = (
        None if best_state is None else _row_numpy(best_state, best_row)
    )
    return GPResult(
        best_fitness=best_fitness,
        best_fitness_history=tuple(best_history),
        median_fitness_history=tuple(median_history),
        best_channels=best_channels,
        best_generation=best_generation,
        generations=int(generations),
        state=state,
    )
