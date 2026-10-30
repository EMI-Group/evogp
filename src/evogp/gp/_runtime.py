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
  :func:`take_rows`, :func:`take_per_row`, :func:`row_values` and the typed
  scalar constructors :func:`i32`, :func:`f32`.

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
    "channels_numpy",
    "index_array",
    "gate_tensor",
    "columns",
    "row_offsets",
    "take_rows",
    "take_per_row",
    "row_values",
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


def unevaluated(pop_size: int) -> "evoxc.Tensor":
    """An all-``-inf`` (unevaluated) fitness vector of ``pop_size`` rows."""
    evoxc_module = evoxc()
    return evoxc_module.full(
        (int(pop_size),), float(UNEVALUATED), dtype=evoxc_module.float32
    )


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
