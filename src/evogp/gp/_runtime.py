"""Shared, internal plumbing of the ``evogp.gp`` operator layer.

The operator modules (:mod:`evogp.gp.selection`, :mod:`evogp.gp.crossover`,
:mod:`evogp.gp.mutation`, :mod:`evogp.gp.variation`) all need the same five
things and nothing here is part of the public surface (``__init__.py`` does not
re-export it):

* **lazy evoxc access** -- :func:`evoxc` delegates to
  :func:`evogp.gp.encoding._evoxc`, so this subtree still imports ``evoxc``
  only at call time and ``import evogp.gp`` keeps working on a plain checkout;
* **key resolution and the executable memo** -- :func:`resolve_key`,
  :func:`executable`, :func:`stats`, :func:`reset`: every operator module owns
  ONE module-level memo dict (one graph per static configuration) in front of
  THE SAME shared :class:`evoxc.CompileCache` that
  :mod:`evogp.gp.encoding` owns, exactly the layering ``init_population`` uses;
* **config resolution** -- :func:`resolve_configs` accepts the same
  ``(GPTreeConfig, GPConfig)`` pair as ``init_population`` and additionally a
  bare :class:`~evogp.gp.config.GPConfig` with an explicit ``gp_config=``-style
  companion, so an operator can be driven from either end;
* **host-side glue** -- :func:`channels_numpy` (a numpy view of the three
  channels), :func:`index_array` (the ``[0, pop_size)``-validated index vector
  the legacy kernel left undefined) and :func:`unevaluated`;
* **trace-time primitives** -- :func:`columns`, :func:`row_offsets`,
  :func:`constant_index`, :func:`take_rows`, :func:`take_per_row`,
  :func:`row_values`, the typed scalar constructors :func:`i32`, :func:`f32` and
  the ``cumsum`` decomposition :func:`cumsum_axis` (the stablehlo/iree emitter
  has no ``cumsum``).

Two evoxc facts drive the last group and are worth stating once here:

``gather`` is numpy ``take``, not ``take_along_axis``.  ``gather(x, idx, axis=a)``
shapes its result as ``x.shape[:a] + idx.shape + x.shape[a + 1:]``, so a
row-wise lookup of a ``[pop, width]`` channel needs the flattened idiom
``gather(reshape(x, (pop * width,)), row * width + column)`` that
:func:`take_per_row` implements (the same trick ``encoding.py`` uses for its
single scatter).

A Python ``int``/``float`` scalar is NOT weakly typed.  ``evoxc`` promotes it to
its natural dtype (``int64``/``float64``), so ``minimum(int32_tensor, 3)`` is an
``int64`` tensor and ``select(bool, int32_tensor, 0)`` an ``int64`` one.  The
channels are ``int32``/``float32``/``int32`` by contract, so any literal that
takes part in *value* arithmetic goes through :func:`i32`/:func:`f32` (a typed
rank-0 constant) and every operator casts its three outputs back to the channel
dtypes before returning.
"""

from __future__ import annotations

import numpy as np

from . import encoding
from .config import GPConfig, GPTreeConfig
from .state import UNEVALUATED, GPState

__all__ = [
    "evoxc",
    "resolve_key",
    "resolve_configs",
    "compile_cache",
    "executable",
    "stats",
    "reset",
    "key_spec",
    "unevaluated",
    "resolve_device",
    "is_tensor",
    "device_text",
    "to_device",
    "channels_numpy",
    "index_array",
    "gate_tensor",
    "columns",
    "row_offsets",
    "constant_index",
    "take_rows",
    "take_per_row",
    "row_values",
    "cumsum_axis",
    "i32",
    "f32",
    "as_channels",
    "OPS",
]


# --------------------------------------------------------------------------- #
# lazy evoxc access, memo and cache
# --------------------------------------------------------------------------- #
def evoxc():
    """The ``evoxc`` module, imported lazily (see ``encoding._evoxc``)."""
    return encoding._evoxc()


def compile_cache():
    """The one shared :class:`evoxc.CompileCache` of this subtree."""
    return encoding._compile_cache()


def resolve_key(key):
    """A concrete ``evoxc`` key from an int seed or an explicit key tensor."""
    return encoding._resolve_key(evoxc(), key)


def key_spec(evoxc_module, key):
    """``TensorSpec`` of a concrete key tensor."""
    return evoxc_module.TensorSpec(
        tuple(int(size) for size in key.shape), key.dtype
    )


def executable(memo: dict, memo_key, factory, specs, backend: str, device: str):
    """The memoized executable of one static configuration.

    Args:
        memo: the operator module's own ``static key -> (fn, executable)`` dict.
        memo_key: the static configuration (config objects, shapes, knobs).
        factory: a zero-argument callable that builds the traced graph closure.
        specs: the ``evoxc.TensorSpec``/static arguments of that closure.
        backend: evoxc backend name.
        device: evoxc device string.

    Returns:
        The ``evoxc`` executable.

    Notes:
        The closure is kept alive next to its executable because
        ``evoxc.build_cached`` memoizes on the callable's identity.
    """
    entry = memo.get(memo_key)
    if entry is not None:
        return entry[1]
    evoxc_module = evoxc()
    fn = factory()
    built = evoxc_module.build_cached(
        fn, *specs, backend=backend, device=device, cache=compile_cache()
    )
    memo[memo_key] = (fn, built)
    return built


def stats(memo: dict) -> dict:
    """Counters of one operator module (its graph memo + the shared cache)."""
    out = {"graphs": len(memo), "compile_cache_size": 0}
    cache = encoding._COMPILE_CACHE
    if cache is not None:
        try:
            out["compile_cache_size"] = len(cache)
        except TypeError:  # pragma: no cover - cache without __len__
            out["compile_cache_size"] = -1
    return out


def reset(memo: dict) -> None:
    """Drop every memoized operator graph (the compile cache is untouched)."""
    memo.clear()


# --------------------------------------------------------------------------- #
# config resolution and host-side glue
# --------------------------------------------------------------------------- #
def resolve_configs(config, gp_config=None, pop_size=None):
    """``(GPTreeConfig, GPConfig, pop_size)`` from the accepted config spellings.

    Accepted:

    * ``(GPTreeConfig, GPConfig)`` -- the same pair ``init_population`` takes
      (then ``gp_config`` must be ``None``);
    * ``GPTreeConfig`` + ``gp_config=GPConfig(...)``;
    * a bare ``GPTreeConfig`` -- ``gp_config`` is then ``None`` in the result
      (the operator takes its own knobs from its arguments) and ``pop_size`` is
      whatever the caller passed (an operator reads it from the state);
    * a bare ``GPConfig`` -- only for the checks that do not need the tree
      description; the tree config is then returned as ``None``.

    Args:
        config: one of the spellings above.
        gp_config: the operator configuration for a bare tree config.
        pop_size: a cross-check (and the source of the population size for a
            bare tree config).

    Raises:
        TypeError: on a wrong config type.
        ValueError: when a pair is malformed or the sizes disagree.
    """
    if isinstance(config, tuple):
        if gp_config is not None:
            raise ValueError(
                "pass either a (GPTreeConfig, GPConfig) pair or "
                "gp_config=GPConfig(...), not both"
            )
        _, pair_pop = encoding._resolve_config_pop(config, None)
        if pop_size is not None and int(pop_size) != pair_pop:
            raise ValueError(
                f"pop_size={pop_size} contradicts the GPConfig's "
                f"pop_size={pair_pop}"
            )
        return config[0], config[1], pair_pop

    if isinstance(config, GPConfig):
        if gp_config is not None:
            raise ValueError(
                "gp_config= is only meaningful together with a GPTreeConfig; "
                "a bare GPConfig already IS the operator configuration"
            )
        if pop_size is not None and int(pop_size) != int(config.pop_size):
            raise ValueError(
                f"pop_size={pop_size} contradicts the GPConfig's "
                f"pop_size={config.pop_size}"
            )
        return None, config, int(config.pop_size)

    if not isinstance(config, GPTreeConfig):
        raise TypeError(
            "config must be a GPTreeConfig, a (GPTreeConfig, GPConfig) pair or "
            f"a GPConfig, got {type(config).__name__}"
        )
    if gp_config is not None and not isinstance(gp_config, GPConfig):
        raise TypeError(
            f"gp_config must be a GPConfig, got {type(gp_config).__name__}"
        )
    resolved_pop = None
    if pop_size is not None:
        resolved_pop = int(pop_size)
    elif gp_config is not None:
        resolved_pop = int(gp_config.pop_size)
    if gp_config is not None and pop_size is not None:
        if int(pop_size) != int(gp_config.pop_size):
            raise ValueError(
                f"pop_size={pop_size} contradicts the GPConfig's "
                f"pop_size={gp_config.pop_size}"
            )
    return config, gp_config, resolved_pop


def channels_numpy(state) -> tuple:
    """``(opcode, payload, size)`` of a state/triple as numpy arrays."""
    return encoding._channel_arrays(state)


def as_channels(state):
    """The three channel tensors of a state (a bare triple passes through)."""
    if isinstance(state, GPState):
        return state.channels()
    try:
        opcode, payload, size = state
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "state must be a GPState or an (opcode, payload, size) triple"
        ) from exc
    return opcode, payload, size


def index_array(value, name: str, pop_size: int) -> np.ndarray:
    """A validated ``int32`` index vector of one population.

    The legacy kernels read out-of-bounds rows when an index was outside
    ``[0, pop_size)`` (undefined behaviour); this raises instead, naming the
    first offending entry.

    Args:
        value: array-like, ``evoxc.Tensor`` or a 1-D sequence of integers.
        name: the argument name used in error messages.
        pop_size: the exclusive upper bound.

    Raises:
        TypeError: for a non-integral dtype.
        ValueError: for a wrong rank, an empty vector, or an out-of-range entry.
    """
    array = np.asarray(encoding._as_numpy(value, name))
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind not in "iu":
        if array.dtype.kind != "f" or not np.all(array == np.floor(array)):
            raise TypeError(
                f"{name} must contain integers, got dtype {array.dtype}"
            )
    array = array.astype(np.int64)
    bad = np.nonzero((array < 0) | (array >= pop_size))[0]
    if bad.size:
        first = int(bad[0])
        raise ValueError(
            f"{name}[{first}]={int(array[first])} is outside [0, pop_size="
            f"{pop_size}) -- the legacy kernel read that row out of bounds, "
            "so it is rejected here"
        )
    return array.astype(np.int32)


def unevaluated(pop_size: int, device=None) -> "evoxc.Tensor":
    """An all-``-inf`` (unevaluated) fitness vector of ``pop_size`` rows.

    Args:
        pop_size: number of rows.
        device: optional device (a spelling like ``"rocm"`` / ``"rocm:0"`` or an
            ``evoxc.Device``).  ``None``/``"cpu"`` keeps the host vector the
            numpy path has always produced; a device spelling uploads it so the
            vector lives where the state's channels live (the driver reads it
            back once per generation anyway).
    """
    evoxc_module = evoxc()
    vector = evoxc_module.full(
        (int(pop_size),), float(UNEVALUATED), dtype=evoxc_module.float32
    )
    target = resolve_device(evoxc_module, device)
    return vector if vector.device == target else vector.to(target)


# --------------------------------------------------------------------------- #
# device plumbing (the iree route is device-resident end to end)
# --------------------------------------------------------------------------- #
def resolve_device(evoxc_module, device):
    """The ``evoxc.Device`` of a device spelling (``None``/``"cpu"`` = host).

    Accepts an ``evoxc.Device`` unchanged, and the ``"<kind>"`` /
    ``"<kind>:<index>"`` spellings the operator entry points take (``"rocm"``
    on this AMD box, ``"cuda"`` on an NVIDIA one, ``"cpu"`` for the numpy
    backend).
    """
    if device is None:
        return evoxc_module.Device("cpu", 0)
    if isinstance(device, evoxc_module.Device):
        return device
    text = str(device).strip()
    kind, _, index = text.partition(":")
    if not kind:
        raise ValueError(f"device must be a non-empty device string, got {device!r}")
    return evoxc_module.Device(kind, int(index) if index else 0)


def is_tensor(value) -> bool:
    """``True`` for a concrete ``evoxc.Tensor`` (a device or host buffer)."""
    return isinstance(value, evoxc().Tensor)


def device_text(device) -> str:
    """A stable ``"kind:index"`` spelling of a device (for memo keys)."""
    resolved = resolve_device(evoxc(), device)
    return f"{resolved.kind}:{resolved.index}"


def to_device(value, device):
    """``value`` on ``device`` (a no-op when it already lives there).

    A numpy array is uploaded with ``evoxc.from_numpy(...).to(device)``; an
    ``evoxc.Tensor`` is moved with ``.to(device)``.  Nothing is copied when the
    tensor is already on the target device, so a device-resident caller pays
    no transfer per generation.
    """
    evoxc_module = evoxc()
    target = resolve_device(evoxc_module, device)
    if isinstance(value, evoxc_module.Tensor):
        return value if value.device == target else value.to(target)
    wrapped = evoxc_module.from_numpy(np.asarray(value))
    return wrapped if wrapped.device == target else wrapped.to(target)


# --------------------------------------------------------------------------- #
# trace-time primitives
# --------------------------------------------------------------------------- #
def i32(evoxc_module, value: int):
    """A typed rank-0 ``int32`` constant (a Python int would promote to 64 bits)."""
    return evoxc_module.constant(
        evoxc_module.full((), int(value), dtype=evoxc_module.int32)
    )


def f32(evoxc_module, value: float):
    """A typed rank-0 ``float32`` constant."""
    return evoxc_module.constant(
        evoxc_module.full((), float(value), dtype=evoxc_module.float32)
    )


def columns(evoxc_module, width: int):
    """``[1, width]`` int32 column indices (broadcast against any row axis)."""
    table = np.arange(int(width), dtype=np.int32).reshape(1, int(width))
    return evoxc_module.constant(evoxc_module.from_numpy(table))


def row_offsets(evoxc_module, pop_size: int, width: int):
    """``[pop_size, 1]`` int64 ``row * width`` offsets for the flattened gathers."""
    table = (np.arange(int(pop_size), dtype=np.int64) * int(width)).reshape(-1, 1)
    return evoxc_module.constant(evoxc_module.from_numpy(table))


def constant_index(evoxc_module, length: int):
    """The constant ``[0, length)`` int32 vector (a static row order).

    The index list of "keep the first ``length`` rows": the offspring rows an
    elite splice keeps, which is static for one graph, so it is baked in here
    rather than passed by the caller.
    """
    table = np.arange(int(length), dtype=np.int32)
    return evoxc_module.constant(evoxc_module.from_numpy(table))


def take_rows(evoxc_module, x, indices):
    """Row selection of a ``[pop_size, ...]`` channel (``gather`` along axis 0)."""
    return evoxc_module.gather(x, evoxc_module.cast(indices, evoxc_module.int32), axis=0)


def take_per_row(evoxc_module, x, index):
    """Row-wise ``take_along_axis`` of a 2-D channel.

    Args:
        x: ``[pop_size, width]`` channel.
        index: ``[pop_size, k]`` integer column indices (clamped by the caller).

    Returns:
        ``[pop_size, k]`` values, ``x[i, index[i, j]]``.

    Notes:
        ``evoxc.gather`` is numpy ``take``: a ``[pop_size, k]`` index tensor
        against ``axis=1`` would produce ``[pop_size, pop_size, k]``.  The
        flattened form below is the per-row lookup the operator layer needs.
    """
    pop_size, width = int(x.shape[0]), int(x.shape[1])
    k = int(index.shape[1])
    flat = evoxc_module.reshape(x, (pop_size * width,))
    flat_index = evoxc_module.reshape(
        evoxc_module.add(row_offsets(evoxc_module, pop_size, width), index),
        (pop_size * k,),
    )
    values = evoxc_module.gather(flat, flat_index, axis=0)
    return evoxc_module.reshape(values, (pop_size, k))


def row_values(evoxc_module, x, index):
    """``[pop_size, 1]`` per-row lookup of a 2-D channel at a 1-D column index."""
    pop_size = int(x.shape[0])
    return take_per_row(
        evoxc_module, x, evoxc_module.reshape(index, (pop_size, 1))
    )


def cumsum_axis(evoxc_module, x, axis: int, zero):
    """``cumsum(x, axis=axis)`` over a STATIC axis, decomposed into primitives.

    Args:
        x: the tensor to scan.
        axis: the scanned axis (its length must be static -- it always is here).
        zero: a typed rank-0 constant of ``x``'s dtype (``i32(evoxc, 0)`` /
            ``f32(evoxc, 0.0)``); it pads the shifted copies, and it must not be
            a Python literal or the concatenation would promote to 64 bits.

    Returns:
        The scanned tensor, same shape and dtype as ``x``.

    Notes:
        The stablehlo emitter that ``backend="iree"`` uses has no ``cumsum``
        ("op 'cumsum' is not supported in v1"), so the scan is built from
        ``slice``/``broadcast``/``concatenate``/``add``: ``cumsum[i] =
        x[i] + x[i-1] + ... + x[0]`` is the running sum of the ``L - 1``
        left-shifted copies of ``x``, each padded with ``zero`` at the front.

        The additions are in the same order as a sequential scan, so the result
        is bit-identical to ``evoxc.cumsum`` on the numpy backend (exactly so
        for the integer channels, which is what the encoding scans).
    """
    rank = len(x.shape)
    length = int(x.shape[axis])
    if length <= 1 or rank == 0:
        return x
    shape = [int(dim) for dim in x.shape]
    acc = x
    for shift in range(1, length):
        # `shifted[j] = x[j - shift]`: keep the FIRST length - shift entries
        # (dropping the tail) and prepend `shift` zeros.
        lengths = list(shape)
        lengths[axis] = length - shift
        part = evoxc_module.slice(
            x, start=(0,) * rank, lengths=tuple(lengths)
        )
        pad_shape = list(shape)
        pad_shape[axis] = shift
        shifted = evoxc_module.concatenate(
            (evoxc_module.broadcast(zero, tuple(pad_shape)), part), axis=axis
        )
        acc = evoxc_module.add(acc, shifted)
    return acc


def gate_tensor(evoxc_module, key, prob, pop_size: int):
    """The per-row ``bool`` filter of a masked operator apply.

    ``prob is None`` (or ``>= 1``) means every row, without drawing from
    ``key``; ``prob <= 0`` means no row.  Otherwise one ``uniform`` draw.
    """
    shape = (int(pop_size),)
    if key is None or prob is None or float(prob) >= 1.0:
        return evoxc_module.constant(evoxc_module.ones(shape, evoxc_module.bool_))
    if float(prob) <= 0.0:
        return evoxc_module.constant(evoxc_module.zeros(shape, evoxc_module.bool_))
    draw = evoxc_module.random.uniform(key, shape, 0.0, 1.0)
    return evoxc_module.less(draw, f32(evoxc_module, prob))


#: The channel dtypes every operator returns (``opcode``, ``payload``, ``size``).
OPS = ("opcode", "payload", "size")
