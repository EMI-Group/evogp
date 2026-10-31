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

The operator work stays in graphs, the evaluation stays on the host
------------------------------------------------------------------
``step`` runs the operator layer's traced graph (one ``evoxc.run``) plus one
small traced graph that splices the elite rows in front of the offspring -- no
host round trip, and no traced ``evaluate``.  The evaluation is a plain Python
call, so the driver's own host-readback budget is **one fitness vector per
generation** (the readback of ``evaluate``'s ``(pop_size,)`` result); the
operator graph stages the channels host-side internally, which is
:mod:`evogp.gp.variation`'s own (pre-existing) behaviour, not a driver sync.

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
* :func:`step` reads that vector back ONCE (the driver's single sync: the
  history/elite bookkeeping needs the numbers), picks the elite rows on the
  host, uploads that small index vector and runs the elite splice as a traced
  graph ON THE DEVICE, so the channels are never downloaded and never
  re-uploaded;
* :func:`run` alternates ``evaluate -> one readback -> variation -> splice`` and
  downloads exactly once more, at the end, to report
  :attr:`GPResult.best_channels` as host rows.

Both fitness kinds are accepted from the SAME ``evaluate`` callable: a numpy
array (the host path, unchanged) or an ``evoxc`` tensor, whose device is then
the run's device.  Nothing has to be declared: an objective that returns a
device tensor drives the device path, and the fitness contract (MAXIMIZED,
non-finite = ``-inf``) is applied on the readback either way.

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

#: ``(static configuration) -> (graph closure, executable)`` of the elite graph.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the driver layer (its elite-graph memo + the shared cache)."""
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop the memoized elite graph (the operator graphs are their own memos)."""
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


def _fitness(state: GPState, evaluate: Callable) -> Tuple[np.ndarray, object]:
    """The host readback of one evaluation (the driver's only sync point).

    Returns:
        ``(values, raw)`` -- the ``[pop_size]`` float32 host fitness (fresh,
        writable, non-finite mapped to ``-inf``) and the object ``evaluate``
        returned, which :func:`_variation_fitness` may hand back to the operator
        layer unchanged when it already lives on the device.

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


def _variation_fitness(raw, values: np.ndarray):
    """The fitness vector handed to :func:`evogp.gp.variation.variation`.

    A device objective just produced a device tensor that the driver read back a
    moment ago, so passing that tensor on saves the ``pop_size``-element upload
    a device generation would otherwise pay.  The substitution is made only when
    the readback is entirely finite: then the device vector and the sanitized
    host vector hold the same values (the in-graph ``non-finite -> -inf`` map is
    a no-op) and the operator layer sees identical input either way.  A
    population that produced a non-finite fitness falls back to the host vector,
    which is what carries the sanitizing, so the operator layer never sees a NaN.
    """
    if not _runtime.is_tensor(raw):
        return values
    if np.dtype(raw.dtype) != np.dtype(np.float32):
        return values
    if tuple(int(dim) for dim in raw.shape) != (int(values.shape[0]),):
        return values
    return raw if bool(np.isfinite(values).all()) else values


def _elite_indices(fitness: np.ndarray, elite_count: int) -> np.ndarray:
    """``[elite_count]`` int32 row indices of the best individuals (best first)."""
    if elite_count <= 0:
        return np.zeros(0, dtype=np.int32)
    # `-fitness` turns the descending search into an ascending `argsort`; the
    # stable kind keeps the lower row index on a tie (an all-`-inf` population
    # therefore keeps rows 0..k-1, exactly like torch's `argmax`).
    order = np.argsort(-fitness, kind="stable")
    return order[: int(elite_count)].astype(np.int32)


def _elite_fn(pop_size: int, width: int):
    """The traced ``(parent, offspring, index) -> channels`` graph of one shape."""

    def operator(p_opcode, p_payload, p_size, o_opcode, o_payload, o_size, index):
        evoxc_module = _runtime.evoxc()

        def pick(parent, offspring):
            # One concatenated [2 * pop_size, width] buffer, then a single row
            # gather: `index` already holds the global row of every output row
            # (elite rows from the first half, kept offspring rows from the
            # second), so the two sources cannot be selected inconsistently.
            merged = evoxc_module.concatenate([parent, offspring], axis=0)
            return evoxc_module.gather(merged, index, axis=0)

        return (
            pick(p_opcode, o_opcode),
            pick(p_payload, o_payload),
            pick(p_size, o_size),
        )

    return operator


def _state_device(state: GPState):
    """The device a state's channels live on, or ``None`` for a host population.

    Reads the tensors' own metadata only (``Tensor.device`` is a host-side field,
    not a sync), which is how the driver decides between the two splice routes:
    a device-resident population never downloads its channels, and a host
    population keeps the byte-for-byte path it has always had.
    """
    channels = _runtime.as_channels(state)
    if not all(_runtime.is_tensor(channel) for channel in channels):
        return None
    device = channels[0].device
    return None if device.kind == "cpu" else device


def _inject_elites(parent: GPState, offspring: GPState, index: np.ndarray,
                   backend: str, device: str,
                   target=None) -> Tuple[object, object, object]:
    """Prepend the selected parent rows to the offspring rows.

    Args:
        parent: the evaluated generation (the elite rows are copied out of it).
        offspring: the operator layer's offspring.
        index: ``[pop_size]`` int32 global row indices -- elite rows from the
            first half of the merged buffer, kept offspring rows from the second.
        backend: evoxc backend name.
        device: evoxc device spelling (the host path passes it through verbatim).
        target: the device of a device-resident population (``None`` on the host
            path).  When set, the index vector is uploaded once and the whole
            splice runs on that device, so neither the channels nor the result
            ever reach the host.
    """
    evoxc_module = _runtime.evoxc()
    pop_size, width = (int(dim) for dim in offspring.opcode.shape)
    parent_channels = parent.channels()
    offspring_channels = offspring.channels()
    if target is not None:
        index = _runtime.to_device(
            np.ascontiguousarray(index, dtype=np.int32), target
        )
        device_key = _runtime.device_text(target)
    else:
        device_key = (
            device if isinstance(device, str) else _runtime.device_text(device)
        )
    memo_key = (
        pop_size,
        width,
        backend,
        device_key,
        tuple(str(tensor.dtype) for tensor in parent_channels),
    )
    specs = tuple(
        evoxc_module.TensorSpec((pop_size, width), tensor.dtype)
        for tensor in parent_channels + offspring_channels
    ) + (evoxc_module.TensorSpec((pop_size,), evoxc_module.int32),)
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _elite_fn(pop_size, width),
        specs,
        backend,
        device if target is None else target,
    )
    return evoxc_module.run(
        built,
        parent.opcode,
        parent.payload,
        parent.size,
        offspring.opcode,
        offspring.payload,
        offspring.size,
        index,
    )


def _advance(config, state: GPState, fitness: np.ndarray, key=None,
             backend: str = "numpy", device: str = "cpu",
             fitness_input=None) -> GPState:
    """The next generation from an already evaluated ``state``.

    Args:
        config: the run's config.
        state: the evaluated population.
        fitness: its host fitness readback (``[pop_size]``, sanitized).
        key: the RNG key to vary with (``None`` continues ``state.key``).
        backend: evoxc backend name.
        device: evoxc device spelling.
        fitness_input: the object ``evaluate`` returned, when it is an ``evoxc``
            tensor -- :func:`_variation_fitness` decides whether it can replace
            the host vector (no upload on a device generation).
    """
    _tree_config, gp_config = _configs(config, population_size(state))
    pop_size = population_size(state)
    elite_count = int(gp_config.elite_count)
    elite_idx = _elite_indices(fitness, elite_count)
    # A device-resident population keeps every channel on the device; a host
    # population takes the host path unchanged.
    target = _state_device(state)

    offspring, next_key = _variation.variation(
        config,
        state,
        _variation_fitness(fitness_input, fitness),
        state.key if key is None else key,
        backend=backend,
        device=device,
    )

    if elite_count:
        # Output row i is parent[elite_idx[i]] for i < elite_count and
        # offspring[i - elite_count] after that; the offspring rows beyond
        # `pop_size - elite_count` are dropped (they are exchangeable draws).
        index = np.concatenate(
            [
                elite_idx.astype(np.int64),
                pop_size + np.arange(pop_size - elite_count, dtype=np.int64),
            ]
        ).astype(np.int32)
        opcode, payload, size = _inject_elites(
            state, offspring, index, backend, device, target
        )
        next_fitness = np.empty(pop_size, dtype=np.float32)
        next_fitness[:elite_count] = fitness[elite_idx]
        next_fitness[elite_count:] = float(UNEVALUATED)
    else:
        opcode, payload, size = offspring.channels()
        next_fitness = np.full(pop_size, float(UNEVALUATED), dtype=np.float32)

    # The elite rows are bit-identical copies of evaluated parents, so their
    # fitness is genuinely known; the offspring are unevaluated until the next
    # `evaluate` call.  A device population gets that vector back on its own
    # device (a `pop_size`-element upload) so the state stays resident.
    if target is not None:
        fitness_vector = _runtime.to_device(next_fitness, target)
    else:
        fitness_vector = _runtime.evoxc().from_numpy(next_fitness)

    return GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=fitness_vector,
        key=next_key,
    )


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
    elite is read out of ``state`` into the *same* graph that assembles the next
    population, so the elite rows are bit-identical copies.

    Args:
        config: a ``(GPTreeConfig, GPConfig)`` pair (or a bare
            ``GPTreeConfig``, then the ``GPConfig`` defaults apply).
        state: the population to evaluate (shapes must match the config).
        evaluate: ``channels -> (pop_size,) float32``, MAXIMIZED, non-finite =
            ``-inf``.  Called exactly once per ``step`` and never traced.  A
            ``SymbolicRegression`` / ``CustomLoss`` instance, its ``.evaluate``,
            its ``.evaluate_graph`` or any engine adapter all qualify.  The
            result may be a numpy array (the host path) or an ``evoxc`` tensor;
            a DEVICE tensor is what makes the whole splice device-resident --
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
        device-resident ``state`` all of these live on the same device and the
        channels are never downloaded -- the only host transfer of the
        generation is the one fitness readback (:func:`_fitness`), whose
        numbers the elite search and the history share.

    Raises:
        ValueError: when ``evaluate`` returns the wrong shape and when the state
            disagrees with the config; ``TypeError`` when ``evaluate`` is not
            callable.
    """
    fitness = _fitness(state, evaluate)
    return _advance(
        config,
        state,
        fitness[0],
        key=key,
        backend=backend,
        device=device,
        fitness_input=fitness[1],
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
    """
    if not isinstance(generations, int) or isinstance(generations, bool):
        raise ValueError(f"generations must be an int, got {generations!r}")
    if generations < 0:
        raise ValueError(f"generations must be >= 0, got {generations}")
    _configs(config, pop_size)

    state = init_gp(
        config, key, pop_size=pop_size, backend=backend, device=device
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
