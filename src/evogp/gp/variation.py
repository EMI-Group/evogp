"""One GP generation of the evoxc GP stack: select, cross, mutate (and elites).

``variation(config, state, fitness, key)`` is the operator-layer counterpart of
``../algorithm/genetic_programming.py::GeneticProgramming.step``: it selects the
parents, applies crossover with ``crossover_prob``, applies mutation with
``mutation_prob`` and -- when ``elite_count`` is given -- keeps that many parent
rows in front of the offspring.  With the default ``elite_count=0`` it returns
**offspring only**, which is what the layer always did; the evaluation stays the
driver's, which is why this module never looks at a problem, an engine or a tree
other than through the channels.

One traced graph per static configuration
-----------------------------------------
The whole generation is ONE graph -- inputs ``(opcode, payload, size, fitness,
key)``, outputs ``(opcode, payload, size, next_fitness, next_key)`` -- memoized
in this module and compiled through the shared :class:`evoxc.CompileCache`, so a
generation costs exactly one ``evoxc.run`` and no host round trip: the fitness
vector of the NEXT state is an in-graph output (``-inf`` for every row when
there is no elite), never an uploaded host allocation.  Only the values vary per
call; ``state``/``fitness``/``key`` supply them.

Elitism is folded into the same graph
-------------------------------------
``elite_count=k`` prepends the ``k`` best parent rows, bit-identically, to the
first ``pop_size - k`` offspring rows INSIDE the graph -- the splice the driver
used to run as a second executable per generation (:file:`workflow.py`).  The
elite rows are ranked in-graph with ``argsort(stable=True)`` on ``-fitness``, so
ties go to the lower row index exactly like
``np.argsort(-fitness, kind="stable")``; a caller that already knows the rows
passes ``elite_index=`` instead.  ``elite_count=0`` (the default) is
offspring-only, i.e. the behaviour this module always had.

Inside the graph (all from the primitives of the sibling modules, so there is no
second implementation of anything):

1. two parent draws from :func:`evogp.gp.selection.select_indices` -- left and
   donor, from two split keys, both by the configured selection mode (classic
   two-parent tournament; the legacy ``DefaultCrossover`` instead drew both
   uniformly from the selection's survivors, which is the same distribution
   after one tournament round);
2. the donor rows and both splice positions from
   :func:`evogp.gp.crossover._draw_positions`, gated by ``crossover_prob``;
3. :func:`evogp.gp.crossover.splice` -- rows that are gated off or would
   overflow ``max_nodes`` stay verbatim copies of their selected parent;
4. :func:`evogp.gp.mutation.apply_mutation` with ``GPConfig.mutation_kind`` and
   ``mutation_prob`` (its own independent per-row gate, like the legacy
   sequential operator pair);
5. with ``elite_count > 0``, :func:`_assemble_elites` -- the elite rows gathered
   out of the parent channels in front of the offspring, plus the next fitness
   vector (elite values, ``-inf`` for offspring).

Determinism: the RNG is the key.  The key is advanced **in-graph** (five
``split_n`` children, the fifth returned as the next key), so ``variation`` is a
pure function of ``(config, state, fitness, key)`` and two calls with the same
key produce bit-identical offspring.

Device-resident generations
---------------------------
Once ``state`` holds concrete ``evoxc`` tensors that live on a device (what
``encoding.init_population(..., backend="iree", device="rocm")`` returns), this
function stays on that device end to end: the channels and the fitness vector go
straight into ``evoxc.run``, the shape/dtype checks read the tensors' metadata
instead of ``np.asarray`` (no host readback, no upload), and the returned state
-- channels, the next fitness vector and next key -- lives on the device too, so
one call is one graph run with ZERO host transfers (generation, elite selection
and splice all happen in that run).  Nothing about the graph changes: the same
traced program serves both routes, and the host (numpy) path is untouched, so
both produce the same offspring for the same seed.

A device run is driven by the tensors' own device; a host (numpy) triple with a
device ``device=`` spelling is uploaded once, and a plain ``"cpu"`` run passes
numpy arrays exactly as it always has.
"""

from __future__ import annotations

import numpy as np

from . import _runtime, crossover as _crossover, encoding, mutation as _mutation
from . import selection as _selection
from .config import GPConfig
from .state import UNEVALUATED, GPState

__all__ = [
    "variation",
    "cache_stats",
    "clear_cache",
]

#: ``(static configuration) -> (graph closure, executable)``.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the variation layer (graph memo + shared compile cache)."""
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop every memoized variation graph."""
    _runtime.reset(_EXECUTABLE_MEMO)


#: The channel dtypes the graph's ``TensorSpec`` pins, in ``channels()`` order.
_CHANNEL_DTYPES = (np.int32, np.float32, np.int32)
_CHANNEL_NAMES = ("opcode", "payload", "size")


def _resident_shape(channels, configured_pop) -> tuple:
    """``(pop_size, width)`` of a device-resident triple, from its metadata.

    The device counterpart of the host-side checks in :func:`variation`: the
    same messages, read off ``Tensor.shape``/``Tensor.dtype`` so a device
    generation never has to download a channel to validate it.

    Raises:
        ValueError: on a wrong rank, disagreeing shapes, an empty population, a
            channel dtype the traced graph does not accept, or a population
            size that contradicts the config.
    """
    _, _, size = channels
    if len(size.shape) != 2:
        raise ValueError(
            f"channels must be 2-D [pop_size, max_nodes], got {tuple(size.shape)}"
        )
    for name, channel, want in zip(_CHANNEL_NAMES, channels, _CHANNEL_DTYPES):
        if np.dtype(channel.dtype) != np.dtype(want):
            raise ValueError(
                f"the device {name} channel must be {np.dtype(want).name}, got "
                f"{np.dtype(channel.dtype).name}"
            )
        if tuple(channel.shape) != tuple(size.shape):
            raise ValueError(
                "the three channels must share one shape, got "
                f"{tuple(channels[0].shape)}, {tuple(channels[1].shape)}, "
                f"{tuple(size.shape)}"
            )
    pop_size, width = (int(dim) for dim in size.shape)
    if pop_size == 0 or width == 0:
        raise ValueError("the population must be non-empty")
    if configured_pop is not None and int(configured_pop) != pop_size:
        raise ValueError(
            f"the population has {pop_size} rows but the config's pop_size is "
            f"{configured_pop}"
        )
    return pop_size, width


def _resident_fitness(fitness, pop_size: int, device):
    """A ``float32 [pop_size]`` fitness tensor on ``device`` (upload if numpy).

    A host array or a host ``evoxc.Tensor`` is moved once, so a caller may keep
    reading the fitness back per generation and still drive a device run.
    """
    if _runtime.is_tensor(fitness):
        shape = tuple(int(dim) for dim in fitness.shape)
        if len(shape) != 1 or shape[0] != int(pop_size):
            raise ValueError(
                f"fitness must be 1-D [pop_size={pop_size}], got shape {shape}"
            )
        if np.dtype(fitness.dtype) != np.dtype(np.float32):
            raise ValueError(
                "fitness must be float32, got "
                f"{np.dtype(fitness.dtype).name}"
            )
        return fitness if fitness.device == device else fitness.to(device)
    values = np.asarray(encoding._as_numpy(fitness, "fitness"))
    if values.ndim != 1 or int(values.shape[0]) != int(pop_size):
        raise ValueError(
            f"fitness must be 1-D [pop_size={pop_size}], got shape {values.shape}"
        )
    return _runtime.to_device(values.astype(np.float32), device)


def _sanitize_fitness(evoxc_module, fitness):
    """In-graph ``non-finite -> -inf`` -- the repo-wide fitness rule.

    A ``NaN`` compares false against everything, so it would corrupt both the
    tournament and the elite ranking.  Applying the rule inside the graph is
    what makes a device-fed generation safe WITHOUT the host readback that
    ``workflow._fitness`` uses to sanitize its own copy.
    """
    shape = (int(fitness.shape[0]),)
    return evoxc_module.select(
        evoxc_module.isnan(fitness),
        evoxc_module.broadcast(
            _runtime.f32(evoxc_module, float(UNEVALUATED)), shape
        ),
        fitness,
    )


def _elite_order(evoxc_module, fitness):
    """``[pop_size]`` int32 row order of ``fitness``, best first.

    Sorting the negated vector ASCENDING with ``stable=True`` is the in-graph
    spelling of ``np.argsort(-fitness, kind="stable")``: a tie keeps the lower
    row index (and ``descending=True`` must not be used -- it flips that order).
    """
    return evoxc_module.cast(
        evoxc_module.argsort(
            evoxc_module.multiply(fitness, _runtime.f32(evoxc_module, -1.0)),
            descending=False,
            stable=True,
        ),
        evoxc_module.int32,
    )


def _assemble_elites(evoxc_module, parents, offspring, fitness, elite_count: int,
                     elite_index=None):
    """Prepend ``elite_count`` parent rows to ``pop_size - elite_count`` offspring.

    Args:
        parents: the three channel tensors of the evaluated generation.
        offspring: the three channel tensors the mutation operator produced.
        fitness: the ``[pop_size]`` parent fitness (already sanitized).
        elite_count: how many parent rows to keep, best first.
        elite_index: a ``[elite_count]`` int32 index tensor of parent rows, or
            ``None`` to rank the rows in-graph.

    Returns:
        ``(opcode, payload, size, next_fitness)``.  The elite rows are gathered
        straight out of the parent tensors, so they are bit-identical copies;
        ``next_fitness`` carries their real fitness and ``-inf`` for every
        offspring row.
    """
    pop_size = int(parents[2].shape[0])
    kept = int(elite_count)
    remaining = pop_size - kept
    if elite_index is None:
        elite_index = evoxc_module.slice(
            _elite_order(evoxc_module, fitness), start=(0,), lengths=(kept,)
        )
    elite_index = evoxc_module.cast(elite_index, evoxc_module.int32)
    keep_offspring = (
        _runtime.constant_index(evoxc_module, remaining) if remaining else None
    )
    channels = []
    for parent, child in zip(parents, offspring):
        elite_rows = evoxc_module.gather(parent, elite_index, axis=0)
        if remaining:
            offspring_rows = evoxc_module.gather(
                child, keep_offspring, axis=0
            )
            elite_rows = evoxc_module.concatenate(
                [elite_rows, offspring_rows], axis=0
            )
        channels.append(elite_rows)
    elite_fitness = evoxc_module.gather(fitness, elite_index, axis=0)
    if remaining:
        elite_fitness = evoxc_module.concatenate(
            [
                elite_fitness,
                evoxc_module.broadcast(
                    _runtime.f32(evoxc_module, float(UNEVALUATED)), (remaining,)
                ),
            ],
            axis=0,
        )
    return channels[0], channels[1], channels[2], elite_fitness


def _variation_fn(tree_config, gp_config: GPConfig, pop_size: int, width: int,
                  elite_count: int = 0, index_input: bool = False):
    """The traced ``(channels, fitness, key[, elite_index]) -> ...`` graph.

    Returns ``(opcode, payload, size, next_fitness, next_key)``: the assembled
    generation (``elite_count`` parent rows in front, then the offspring), the
    fitness vector that goes with it (elite values, ``-inf`` for offspring) and
    the advanced key.

    ``index_input`` adds the ``[elite_count]`` parent row indices as a sixth
    argument; without it (and with ``elite_count > 0``) the graph ranks the rows
    itself, so a device generation needs no host round trip to pick them.
    """
    mode = gp_config.selection
    tournament_size = int(gp_config.tournament_size)
    pressure = float(gp_config.selection_pressure)
    crossover_prob = float(gp_config.crossover_prob)
    mutation_prob = float(gp_config.mutation_prob)
    mutation_kind = gp_config.mutation_kind
    kept = int(elite_count)

    def operator(opcode, payload, size, fitness, key, elite_index=None):
        evoxc_module = _runtime.evoxc()
        k_left, k_right, k_cross, k_mutate, k_next = evoxc_module.random.split_n(
            key, 5
        )
        # One in-graph `non-finite -> -inf` pass: the tournament and the elite
        # ranking then see the same vector the host readback would have held.
        fitness = _sanitize_fitness(evoxc_module, fitness)
        parents = _selection.select_indices(
            evoxc_module, fitness, k_left, pop_size, mode, tournament_size, pressure
        )
        donors = _selection.select_indices(
            evoxc_module, fitness, k_right, pop_size, mode, tournament_size, pressure
        )
        left = (
            _runtime.take_rows(evoxc_module, opcode, parents),
            _runtime.take_rows(evoxc_module, payload, parents),
            _runtime.take_rows(evoxc_module, size, parents),
        )
        donor = (
            _runtime.take_rows(evoxc_module, opcode, donors),
            _runtime.take_rows(evoxc_module, payload, donors),
            _runtime.take_rows(evoxc_module, size, donors),
        )
        k_gate, k_position = evoxc_module.random.split_n(k_cross, 2)
        pos, donor_pos = _crossover._draw_positions(
            evoxc_module, k_position, left[2], donor[2]
        )
        gate = _runtime.gate_tensor(evoxc_module, k_gate, crossover_prob, pop_size)
        offspring = _crossover._splice_traced(
            evoxc_module, left, donor, pos, donor_pos, gate
        )
        mutated = _mutation.apply_mutation(
            evoxc_module, tree_config, offspring, k_mutate, mutation_prob,
            mutation_kind,
        )
        if not kept:
            # No elite: every row is a fresh offspring, so the whole next fitness
            # vector is `-inf` -- built here instead of uploaded by the caller.
            next_fitness = evoxc_module.broadcast(
                _runtime.f32(evoxc_module, float(UNEVALUATED)), (pop_size,)
            )
            return mutated[0], mutated[1], mutated[2], next_fitness, k_next
        assembled = _assemble_elites(
            evoxc_module,
            (opcode, payload, size),
            mutated,
            fitness,
            kept,
            elite_index if index_input else None,
        )
        return assembled[0], assembled[1], assembled[2], assembled[3], k_next

    return operator


def _resolve_elites(elite_count, elite_index, pop_size: int):
    """``(elite_count, index_array)`` of one generation, validated.

    ``elite_index`` (when given) fixes ``elite_count`` to its length; a
    contradictory pair is rejected rather than silently preferred, so a caller
    can never splice ``k`` rows while believing it asked for ``j``.

    Raises:
        ValueError: on a negative / oversized ``elite_count`` or a disagreeing,
            out-of-range ``elite_index``.
        TypeError: on a non-integral ``elite_index``.
    """
    if elite_count is None:
        elite_count = 0
    if isinstance(elite_count, bool) or not isinstance(elite_count, (int, np.integer)):
        raise ValueError(f"elite_count must be an int, got {elite_count!r}")
    count = int(elite_count)
    index_array = None
    if elite_index is not None:
        index_array = _runtime.index_array(elite_index, "elite_index", pop_size)
        if count and count != int(index_array.size):
            raise ValueError(
                f"elite_count={count} contradicts {int(index_array.size)} "
                "elite_index entries"
            )
        count = int(index_array.size)
    if count < 0:
        raise ValueError(f"elite_count must be >= 0, got {count}")
    if count > int(pop_size):
        raise ValueError(
            f"elite_count={count} exceeds the population of {int(pop_size)} rows"
        )
    return count, index_array


def _variation_specs(evoxc_module, pop_size: int, width: int, key_tensor,
                     elite_count: int, index_array):
    """The ``TensorSpec`` list of one variation graph."""
    specs = [
        evoxc_module.TensorSpec((int(pop_size), int(width)), evoxc_module.int32),
        evoxc_module.TensorSpec((int(pop_size), int(width)), evoxc_module.float32),
        evoxc_module.TensorSpec((int(pop_size), int(width)), evoxc_module.int32),
        evoxc_module.TensorSpec((int(pop_size),), evoxc_module.float32),
        _runtime.key_spec(evoxc_module, key_tensor),
    ]
    if index_array is not None:
        specs.append(
            evoxc_module.TensorSpec((int(elite_count),), evoxc_module.int32)
        )
    return tuple(specs)


def variation(config, state, fitness, key=0, *, elite_count: int = 0,
              elite_index=None, backend: str = "numpy", device: str = "cpu"):
    """One generation of offspring: select, crossover, mutate (+ elites).

    Args:
        config: a ``(GPTreeConfig, GPConfig)`` pair (the normal case, like
            ``init_population``), or a bare
            :class:`~evogp.gp.config.GPTreeConfig` -- then the operator defaults
            of :class:`~evogp.gp.config.GPConfig` are used.  A bare ``GPConfig``
            is not enough: the mutation operator needs the function set, the
            constant table and ``input_len``.
        state: the parent population (a :class:`~evogp.gp.state.GPState` or an
            ``(opcode, payload, size)`` triple).  Either three host arrays /
            host tensors, or three concrete ``evoxc`` tensors on a device --
            then the whole generation runs device-resident (no readback, no
            upload, a device offspring state).
        fitness: ``float32 [pop_size]`` fitness of the parents (MAXIMIZED,
            ``-inf`` = unevaluated) -- the host readback of the last evaluation.
            A numpy array or an ``evoxc.Tensor``; on a device run a host array
            is uploaded once (see :func:`_resident_fitness`), a device tensor
            goes straight in.
        key: int seed or a concrete key tensor (the state's own key is a valid
            choice: ``variation(config, state, fitness, state.key)``).
        elite_count: how many parent rows to keep, best first, in front of the
            offspring (``0``, the default, is pure offspring).  The rows are
            ranked IN-GRAPH -- ``argsort`` of ``-fitness`` with ``stable=True``,
            i.e. exactly ``np.argsort(-fitness, kind="stable")[:elite_count]``
            -- and copied bit-identically, with their fitness, into the returned
            state (``-inf`` for the offspring rows).
        elite_index: an explicit ``[elite_count]`` vector of parent row indices
            to use instead of the in-graph ranking (it also fixes
            ``elite_count`` to its length); uploaded once on a device run.
        backend: evoxc backend name (``"numpy"`` default; ``"iree"`` for a
            device box).
        device: evoxc device spelling (``"cpu"`` default, ``"rocm"`` /
            ``"rocm:0"`` on this AMD box) or an ``evoxc.Device``.  A
            device-resident ``state`` overrides it, since the tensors' own
            device is the ground truth.

    Returns:
        ``(offspring, next_key)`` -- the next generation
        :class:`~evogp.gp.state.GPState` (channel dtypes of the encoding, the
        fitness vector that goes with those rows, ``key`` set to ``next_key``)
        and that next key (the same object, returned separately because the
        objective's contract is ``variation(...) -> (offspring_state,
        next_key)``).  Both live on the device of ``state``.

    Raises:
        ValueError: on a non-2-D channel, a fitness vector of the wrong length,
            an ``elite_count``/``elite_index`` that does not fit the population,
            or a population that disagrees with the config's ``pop_size``.
        TypeError: on an unsupported config or fitness type.
    """
    tree_config, gp_config, configured_pop = _runtime.resolve_configs(config)
    if tree_config is None:
        raise ValueError(
            "variation needs a GPTreeConfig (the mutation operator draws from "
            "the function set, the constant table and input_len)"
        )
    evoxc_module = _runtime.evoxc()
    channels = _runtime.as_channels(state)
    resident = all(_runtime.is_tensor(channel) for channel in channels)

    if resident:
        pop_size, width = _resident_shape(channels, configured_pop)
        target = channels[0].device
        channel_inputs = channels
        fitness_input = _resident_fitness(fitness, pop_size, target)
        fitness_device = target
    else:
        opcode_array, payload_array, size_array = _runtime.channels_numpy(state)
        if size_array.ndim != 2:
            raise ValueError(
                f"channels must be 2-D [pop_size, max_nodes], got {size_array.shape}"
            )
        if not (opcode_array.shape == payload_array.shape == size_array.shape):
            raise ValueError(
                "the three channels must share one shape, got "
                f"{opcode_array.shape}, {payload_array.shape}, {size_array.shape}"
            )
        pop_size, width = (int(dim) for dim in size_array.shape)
        if pop_size == 0 or width == 0:
            raise ValueError("the population must be non-empty")
        if configured_pop is not None and int(configured_pop) != pop_size:
            raise ValueError(
                f"the population has {pop_size} rows but the config's pop_size is "
                f"{configured_pop}"
            )
        values = np.asarray(encoding._as_numpy(fitness, "fitness"))
        if values.ndim != 1 or int(values.shape[0]) != pop_size:
            raise ValueError(
                f"fitness must be 1-D [pop_size={pop_size}], got shape {values.shape}"
            )
        values = values.astype(np.float32)
        target = _runtime.resolve_device(evoxc_module, device)
        if target.kind == "cpu":
            # The host path, byte for byte: numpy arrays straight into `run`.
            channel_inputs = (
                opcode_array.astype(np.int32),
                payload_array.astype(np.float32),
                size_array.astype(np.int32),
            )
            fitness_input = values
        else:
            # A host triple aimed at a device backend: one upload, then the run.
            channel_inputs = (
                _runtime.to_device(opcode_array.astype(np.int32), target),
                _runtime.to_device(payload_array.astype(np.float32), target),
                _runtime.to_device(size_array.astype(np.int32), target),
            )
            fitness_input = _runtime.to_device(values, target)
        fitness_device = target

    if gp_config is None:
        gp_config = GPConfig(pop_size=pop_size)

    elite_count, index_array = _resolve_elites(
        elite_count, elite_index, pop_size
    )
    key_tensor = _runtime.to_device(_runtime.resolve_key(key), fitness_device)
    memo_key = (
        tree_config,
        gp_config,
        pop_size,
        width,
        elite_count,
        index_array is not None,
        backend,
        device if isinstance(device, str) else _runtime.device_text(device),
        tuple(int(size_) for size_ in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _variation_fn(
            tree_config, gp_config, pop_size, width, elite_count,
            index_array is not None,
        ),
        _variation_specs(
            evoxc_module, pop_size, width, key_tensor, elite_count, index_array
        ),
        backend,
        device,
    )
    inputs = [
        channel_inputs[0],
        channel_inputs[1],
        channel_inputs[2],
        fitness_input,
        key_tensor,
    ]
    if index_array is not None:
        inputs.append(_runtime.to_device(index_array, fitness_device))
    opcode, payload, size, next_fitness, next_key = evoxc_module.run(
        built, *inputs
    )
    offspring = GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=next_fitness,
        key=next_key,
    )
    return offspring, next_key
