"""One GP generation of the evoxc GP stack: select, cross, mutate -- offspring only.

``variation(config, state, fitness, key)`` is the operator-layer counterpart of
``../algorithm/genetic_programming.py::GeneticProgramming.step`` *minus*
elitism: it selects the parents, applies crossover with ``crossover_prob``,
applies mutation with ``mutation_prob`` and returns **offspring only**.  The
driver owns the elite slice (``GPConfig.elite_count``) and the evaluation, which
is why this module never looks at a problem, an engine or a tree other than
through the channels.

One traced graph per static configuration
-----------------------------------------
The whole generation is ONE graph -- inputs ``(opcode, payload, size, fitness,
key)``, outputs ``(opcode, payload, size, next_key)`` -- memoized in this module
and compiled through the shared :class:`evoxc.CompileCache`, so a generation
costs one ``evoxc.run`` and one host sync (the fitness readback of the next
evaluation).  Only the values vary per call; ``state``/``fitness``/``key``
supply them.

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
   sequential operator pair).

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
-- channels, ``-inf`` fitness and next key -- lives on the device too.  Nothing
about the graph changes: the same traced program serves both routes, and the
host (numpy) path is untouched, so both produce the same offspring for the same
seed.

A device run is driven by the tensors' own device; a host (numpy) triple with a
device ``device=`` spelling is uploaded once, and a plain ``"cpu"`` run passes
numpy arrays exactly as it always has.
"""

from __future__ import annotations

import numpy as np

from . import _runtime, crossover as _crossover, encoding, mutation as _mutation
from . import selection as _selection
from .config import GPConfig
from .state import GPState

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


def _variation_fn(tree_config, gp_config: GPConfig, pop_size: int, width: int):
    """The traced ``(channels, fitness, key) -> channels + next key`` graph."""
    mode = gp_config.selection
    tournament_size = int(gp_config.tournament_size)
    pressure = float(gp_config.selection_pressure)
    crossover_prob = float(gp_config.crossover_prob)
    mutation_prob = float(gp_config.mutation_prob)
    mutation_kind = gp_config.mutation_kind

    def operator(opcode, payload, size, fitness, key):
        evoxc_module = _runtime.evoxc()
        k_left, k_right, k_cross, k_mutate, k_next = evoxc_module.random.split_n(
            key, 5
        )
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
        return mutated[0], mutated[1], mutated[2], k_next

    return operator


def variation(config, state, fitness, key=0, *, backend: str = "numpy",
              device: str = "cpu"):
    """One generation of offspring: select, crossover, mutate.

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
            is uploaded once (see :func:`_resident_fitness`).
        key: int seed or a concrete key tensor (the state's own key is a valid
            choice: ``variation(config, state, fitness, state.key)``).
        backend: evoxc backend name (``"numpy"`` default; ``"iree"`` for a
            device box).
        device: evoxc device spelling (``"cpu"`` default, ``"rocm"`` /
            ``"rocm:0"`` on this AMD box) or an ``evoxc.Device``.  A
            device-resident ``state`` overrides it, since the tensors' own
            device is the ground truth.

    Returns:
        ``(offspring, next_key)`` -- the offspring
        :class:`~evogp.gp.state.GPState` (channel dtypes of the encoding, an
        all-``-inf`` fitness vector, ``key`` set to ``next_key``) and that next
        key (the same object, returned separately because the objective's
        contract is ``variation(...) -> (offspring_state, next_key)``).  Both
        live on the device of ``state``.

    Raises:
        ValueError: on a non-2-D channel, a fitness vector of the wrong length,
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

    key_tensor = _runtime.to_device(_runtime.resolve_key(key), fitness_device)
    memo_key = (
        tree_config,
        gp_config,
        pop_size,
        width,
        backend,
        device if isinstance(device, str) else _runtime.device_text(device),
        tuple(int(size_) for size_ in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _variation_fn(tree_config, gp_config, pop_size, width),
        (
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.float32),
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            evoxc_module.TensorSpec((pop_size,), evoxc_module.float32),
            _runtime.key_spec(evoxc_module, key_tensor),
        ),
        backend,
        device,
    )
    opcode, payload, size, next_key = evoxc_module.run(
        built,
        channel_inputs[0],
        channel_inputs[1],
        channel_inputs[2],
        fitness_input,
        key_tensor,
    )
    offspring = GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=_runtime.unevaluated(pop_size, device=fitness_device),
        key=next_key,
    )
    return offspring, next_key
