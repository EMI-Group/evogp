"""The channel encoding of the evoxc GP stack: generation, validation and the
bridge to the tree layer's ``(value, type, size)`` triple.

Layout (see ``DESIGN.md`` and :mod:`evogp.gp.state`): three parallel
``[pop_size, max_nodes]`` channels in **prefix (pre-order)** order, node 0 =
root, padding past the row's live node count.

``opcode`` int32 (function id or ``NType`` leaf marker), ``payload`` float32
(const value / variable index), ``size`` int32 (subtree node count).
Child addressing is the repository-wide invariant ``c1 = i + 1``,
``c2 = c1 + size[c1]``, ``c3 = c2 + size[c2]``, and a leaf is exactly a node
with ``size == 1``.

Public surface
--------------
``init_population(config, key, ...)``
    One traced, memoized graph per config draws a whole random population from
    the key (the same decision structure as ``../tree/generation.py``: level-wise
    sampling on the padded complete-tree layout, then one scatter into pre-order)
    and returns a :class:`~evogp.gp.state.GPState` whose ``key`` is the key of
    the next generation.
``live_mask(channels)`` / ``row_length(channels)`` / ``is_leaf(size)``
    Derived, never-stored views (host-side numpy).
``validate_channels(channels, input_len=None)``
    The host-side counterpart of the tree layer's decode validation: raises
    ``ValueError`` naming the row and the node (subtree size 0, a subtree past
    ``max_nodes``, a child index outside its parent's subtree, a prefix-order
    violation, an opcode/arity disagreement).
``to_legacy_triple(channels)`` / ``from_legacy_triple(value, type, size)``
    The numpy<->numpy bridge to ``../tree/``'s ``(batch_node_value float32,
    batch_node_type int16, batch_subtree_size int16)`` triple, so the existing
    evoxc engines can evaluate what this stack produces.  Exact round-trip for
    single-output trees.

This module imports **no torch** and -- so that ``import evogp.gp.encoding``
works on a plain checkout -- it also imports ``evoxc`` lazily: every entry point
resolves it through :func:`_evoxc`, which raises a clear ``RuntimeError`` when
the package is missing instead of breaking the import.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from ._constants import FUNC_ARITY, FUNC_TYPE, FUNCS_NAMES, Func, NType
from .config import GPConfig, GPTreeConfig
from .state import GPState

__all__ = [
    "init_population",
    "live_mask",
    "row_length",
    "is_leaf",
    "validate_channels",
    "to_legacy_triple",
    "from_legacy_triple",
    "cache_stats",
    "clear_cache",
    "CACHE_SIZE_ENV",
]

#: Environment variable holding the size of the graph/compile cache.
CACHE_SIZE_ENV = "EVOGP_GP_COMPILE_CACHE_SIZE"
_DEFAULT_CACHE_SIZE = 64


# --------------------------------------------------------------------------- #
# lazy evoxc access
# --------------------------------------------------------------------------- #
def _evoxc():
    """The ``evoxc`` module (imported lazily, see the module docstring)."""
    try:
        import evoxc  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "evogp.gp needs the evoxc package (EvoX Compiler) to build or run a "
            "graph; put the evoxc checkout on PYTHONPATH (e.g. "
            "PYTHONPATH=src:/root/evoxc). Importing evogp.gp itself does not "
            "require evoxc."
        ) from exc
    return evoxc


#: ``(config, pop_size, backend, device, key shape, key dtype) -> (fn, executable)``.
_EXECUTABLE_MEMO: dict = {}
_COMPILE_CACHE = None


def _cache_size() -> int:
    raw = os.environ.get(CACHE_SIZE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_CACHE_SIZE
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CACHE_SIZE_ENV} must be an integer, got {raw!r}"
        ) from exc
    if size <= 0:
        raise ValueError(f"{CACHE_SIZE_ENV} must be positive, got {size}")
    return size


def _compile_cache():
    global _COMPILE_CACHE
    if _COMPILE_CACHE is None:
        _COMPILE_CACHE = _evoxc().CompileCache(maxsize=_cache_size())
    return _COMPILE_CACHE


def cache_stats() -> dict:
    """Counters of the generation layer (graph memo + evoxc compile cache)."""
    stats = {
        "graphs": len(_EXECUTABLE_MEMO),
        "compile_cache_size": 0,
        "compile_cache_maxsize": _cache_size(),
    }
    cache = _COMPILE_CACHE
    if cache is not None:
        try:
            stats["compile_cache_size"] = len(cache)
        except TypeError:  # pragma: no cover - cache without __len__
            stats["compile_cache_size"] = -1
    return stats


def clear_cache() -> None:
    """Drop every memoized generation graph and the evoxc compile cache."""
    global _COMPILE_CACHE
    _EXECUTABLE_MEMO.clear()
    if _COMPILE_CACHE is not None:
        _COMPILE_CACHE.clear()
    _COMPILE_CACHE = None


# --------------------------------------------------------------------------- #
# host-side channel helpers (numpy; the tree layer's decode counterpart)
# --------------------------------------------------------------------------- #
def _as_numpy(value, name: str) -> np.ndarray:
    """A concrete numpy view of a channel (``evoxc.Tensor`` or array-like).

    A tensor that lives on a device is downloaded first (``.numpy()`` refuses a
    device buffer), so the host-side helpers -- :func:`validate_channels`,
    :func:`row_length`, the legacy bridge -- also accept the device-resident
    channels of an ``iree`` run.  That download is why the operator layer never
    routes a hot path through here.
    """
    if isinstance(value, np.ndarray):
        return value
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        device = getattr(value, "device", None)
        if device is not None and getattr(device, "kind", "cpu") != "cpu":
            value = value.to(_evoxc().Device("cpu", 0))
        return np.asarray(value.numpy())
    try:
        return np.asarray(value)
    except Exception as exc:  # noqa: BLE001 - re-raised with the field name
        raise TypeError(
            f"evogp.gp: {name} must be a numpy array or an evoxc.Tensor, got "
            f"{type(value).__name__}"
        ) from exc


def _channel_arrays(channels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(opcode, payload, size)`` as numpy arrays; a state or a triple.

    This is the ONE channel parser of the stack, so every consumer
    (:func:`validate_channels`, :func:`to_legacy_triple`, :func:`row_length`,
    :func:`live_mask`, the evaluator's ``_prepare_channels`` and the problem
    layer's ``pop_size`` read) accepts the same forms: a
    :class:`~evogp.gp.state.GPState`, the three ``[pop_size, max_nodes]``
    channels of a population, or the three 1-D channel rows of a single
    individual (what ``GPResult.best_channels`` holds).  A 1-D triple of equal
    length is promoted to a one-row population ``[1, max_nodes]``; anything else
    is returned as given, for the caller's own rank/shape check.
    """
    if isinstance(channels, GPState):
        opcode, payload, size = channels.channels()
    else:
        try:
            opcode, payload, size = channels
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "evogp.gp: channels must be a GPState or an "
                "(opcode, payload, size) triple"
            ) from exc
    opcode = _as_numpy(opcode, "opcode")
    payload = _as_numpy(payload, "payload")
    size = _as_numpy(size, "size")
    if opcode.ndim == 1 and payload.ndim == 1 and size.ndim == 1:
        if opcode.shape != payload.shape or opcode.shape != size.shape:
            raise ValueError(
                "the three channels of one individual must share one length, "
                f"got opcode {opcode.shape}, payload {payload.shape}, "
                f"size {size.shape}"
            )
        return opcode[None, :], payload[None, :], size[None, :]
    return opcode, payload, size


def is_leaf(size) -> np.ndarray:
    """Live-node mask of the leaves (a leaf's subtree size is exactly 1).

    Accepts a numpy array or an ``evoxc.Tensor``.
    """
    return _as_numpy(size, "size") == 1


def row_length(channels) -> np.ndarray:
    """Real length (node count) of every row: ``size[:, 0]``.

    Host helper: calling it on an ``evoxc.Tensor`` reads the tensor back.
    """
    _, _, size = _channel_arrays(channels)
    return size[:, 0]


def live_mask(channels) -> np.ndarray:
    """``[pop_size, max_nodes]`` bool mask of the real (non-padding) cells.

    Derived, never stored: ``arange(max_nodes)[None, :] < size[:, 0:1]``.
    """
    _, _, size = _channel_arrays(channels)
    if size.ndim != 2:
        raise ValueError(f"size must be 2-D, got shape {size.shape}")
    length = size[:, 0]
    cols = np.arange(size.shape[1])
    return cols[None, :] < length[:, None]


def _first_bad_row(size: np.ndarray) -> Optional[int]:
    """Row index of the first row whose length is not in ``(0, width]``."""
    length = size[:, 0]
    bad = np.nonzero((length <= 0) | (length > size.shape[1]))[0]
    return int(bad[0]) if bad.size else None


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_channels(
    channels,
    input_len: Optional[int] = None,
) -> None:
    """Validate a population encoding, raising ``ValueError`` on any defect.

    The counterpart of the host-side decode validation of the tree layer
    (``../tree/evoxc_engine.py``): a malformed tree is rejected here with the
    row and the node in the message instead of corrupting an evaluation.

    Checks, per row: the row length ``size[0]`` is in ``(0, max_nodes]``; every
    live node has a positive subtree size; a subtree never extends past the row
    length; a leaf (``size == 1``) carries a leaf opcode (``NType.VAR`` /
    ``NType.CONST``); a function node carries a valid function id whose arity
    matches the children the ``size`` channel describes; the children are
    contiguous and the subtree sizes add up (prefix-order violation otherwise);
    and every live cell is claimed by exactly one subtree.

    Args:
        channels: a :class:`~evogp.gp.state.GPState`, an
            ``(opcode, payload, size)`` triple of ``[pop_size, max_nodes]``
            numpy arrays / evoxc tensors, or the three 1-D channel rows of a
            single individual (``GPResult.best_channels``), which is validated
            as a one-row population.
        input_len: when given, ``VAR`` payloads must be integral and in
            ``[0, input_len)``.

    Raises:
        ValueError: naming the offending row and node.
    """
    opcode, payload, size = _channel_arrays(channels)
    if size.ndim != 2:
        raise ValueError(
            "channels must be 2-D [pop_size, max_nodes] (or the three 1-D "
            f"channel rows of one individual), got {size.shape}"
        )
    if opcode.shape != size.shape or payload.shape != size.shape:
        raise ValueError(
            "the three channels must share one shape, got opcode "
            f"{opcode.shape}, payload {payload.shape}, size {size.shape}"
        )
    pop_size, width = size.shape
    if pop_size == 0 or width == 0:
        raise ValueError(f"channels must be non-empty, got shape {size.shape}")

    for row in range(pop_size):
        _validate_row(
            row,
            opcode[row].astype(np.int64),
            payload[row],
            size[row].astype(np.int64),
            width,
            input_len,
        )


def _validate_row(
    row: int,
    opcode: np.ndarray,
    payload: np.ndarray,
    size: np.ndarray,
    width: int,
    input_len: Optional[int],
) -> None:
    total = int(size[0])
    if total <= 0:
        raise ValueError(
            f"row {row} node 0: subtree size {total} is not positive -- the row "
            "is empty (a tree has at least one node)"
        )
    if total > width:
        raise ValueError(
            f"row {row} node 0: subtree size {total} exceeds max_nodes={width}"
        )

    if input_len is not None:
        variables = np.nonzero(
            (size[:total] == 1) & (opcode[:total] == NType.VAR)
        )[0]
        bad = variables[
            (payload[:total][variables] != np.floor(payload[:total][variables]))
            | (payload[:total][variables] < 0)
            | (payload[:total][variables] >= input_len)
        ]
        if bad.size:
            node = int(bad[0])
            raise ValueError(
                f"row {row} node {node}: variable index {payload[node]!r} is not "
                f"an integer in [0, input_len={input_len})"
            )

    seen = np.zeros(total, dtype=bool)
    stack = [0]
    while stack:
        index = stack.pop()
        if index >= total:  # pragma: no cover - defensive, guarded below
            raise ValueError(
                f"row {row} node {index}: child index is outside the row length "
                f"{total}"
            )
        if seen[index]:
            raise ValueError(
                f"row {row} node {index}: claimed by more than one subtree -- "
                "prefix-order violation"
            )
        seen[index] = True

        subtree = int(size[index])
        if subtree <= 0:
            raise ValueError(
                f"row {row} node {index}: subtree size {subtree} is not positive"
            )
        if index + subtree > total:
            raise ValueError(
                f"row {row} node {index}: subtree of size {subtree} runs past "
                f"the row length {total}"
            )
        node_opcode = int(opcode[index])
        if subtree == 1:
            if node_opcode not in (NType.VAR, NType.CONST):
                raise ValueError(
                    f"row {row} node {index}: a leaf (subtree size 1) must carry "
                    f"NType.VAR({NType.VAR}) or NType.CONST({NType.CONST}), got "
                    f"opcode {node_opcode}"
                )
            continue
        if not 0 <= node_opcode < Func.END:
            raise ValueError(
                f"row {row} node {index}: opcode {node_opcode} is neither a leaf "
                f"marker nor a function id in [0, {Func.END})"
            )
        arity = FUNC_ARITY[node_opcode]
        if subtree < 1 + arity:
            raise ValueError(
                f"row {row} node {index}: subtree size {subtree} is too small for "
                f"a {FUNCS_NAMES[node_opcode]} (arity {arity}) node"
            )
        child = index + 1
        parts = 1
        for _ in range(arity):
            if child >= index + subtree:
                raise ValueError(
                    f"row {row} node {index}: child index {child} falls outside "
                    f"the subtree [{index}, {index + subtree})"
                )
            child_size = int(size[child])
            if child_size <= 0:
                raise ValueError(
                    f"row {row} node {child}: subtree size {child_size} is not "
                    "positive"
                )
            parts += child_size
            stack.append(child)
            child += child_size
        if child != index + subtree:
            raise ValueError(
                f"row {row} node {index}: subtree size {subtree} disagrees with "
                f"its children ({parts} nodes) -- prefix-order violation"
            )

    if not seen.all():
        index = int(np.argmin(seen))
        raise ValueError(
            f"row {row} node {index}: not covered by any subtree -- "
            "prefix-order violation"
        )


# --------------------------------------------------------------------------- #
# bridge to the tree layer's (value, type, size) triple
# --------------------------------------------------------------------------- #
def to_legacy_triple(channels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(batch_node_value float32, batch_node_type int16, batch_subtree_size int16)``.

    The exact layout the tree layer's ``Forest`` stores
    (``../tree/generation.py``): on a function node the *value* is the function
    id as a float and the *type* is the ``NType`` marker; on a leaf the value is
    the payload and the type is ``NType.VAR``/``NType.CONST``; the padding is
    zero in all three.

    A single individual (the three 1-D channel rows of a
    ``GPResult.best_channels``) is converted as a one-row population, i.e. the
    result is the same ``[1, max_nodes]`` triple the 2-D form gives.

    Raises:
        NotImplementedError: for an ``OUT_NODE``-tagged opcode (multi-output
            trees pack ``(index << 16) | function_id`` into the value, which
            this encoding does not produce yet).
    """
    opcode, payload, size = _channel_arrays(channels)
    if np.any((opcode & NType.OUT_NODE) != 0):
        raise NotImplementedError(
            "to_legacy_triple: multi-output (OUT_NODE-tagged) trees are not "
            "implemented in the evoxc GP stack yet; build the population with "
            "output_len=1"
        )

    function = size > 1
    illegal = function & ((opcode < 0) | (opcode >= Func.END))
    if illegal.any():
        flat = int(np.nonzero(illegal)[0][0])
        row, node = divmod(flat, opcode.shape[1])
        raise ValueError(
            f"to_legacy_triple: row {row} node {node} carries the function id "
            f"{int(opcode.flat[flat])}, outside [0, {Func.END})"
        )
    clipped = np.where(function, opcode, 0).astype(np.int32)
    legacy_type = np.where(
        size == 0,
        0,
        np.where(function, np.asarray(FUNC_TYPE, np.int32)[clipped], opcode),
    ).astype(np.int16)
    legacy_value = np.where(
        size == 0,
        np.float32(0.0),
        np.where(function, clipped.astype(np.float32), payload.astype(np.float32)),
    ).astype(np.float32)
    legacy_size = np.where(size == 0, 0, size).astype(np.int16)
    return legacy_value, legacy_type, legacy_size


def from_legacy_triple(
    value,
    type,  # noqa: A002 - the tree layer's name for the channel
    size,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(opcode int32, payload float32, size int32)`` from a legacy triple.

    The inverse of :func:`to_legacy_triple`; ``value`` on a function node is
    not part of the encoding and is dropped (the payload of a function node is
    always 0.0), which is exactly what makes the round-trip exact for a
    population this stack produced.

    Raises:
        NotImplementedError: for an ``OUT_NODE``-tagged type (multi-output).
        ValueError: for a type marker outside ``{VAR, CONST, UFUNC, BFUNC,
            TFUNC}``.
    """
    legacy_value = _as_numpy(value, "value").astype(np.float32)
    legacy_type = _as_numpy(type, "type").astype(np.int32)
    legacy_size = _as_numpy(size, "size").astype(np.int32)
    if legacy_value.shape != legacy_size.shape or legacy_type.shape != legacy_size.shape:
        raise ValueError(
            "the legacy triple must share one shape, got value "
            f"{legacy_value.shape}, type {legacy_type.shape}, size {legacy_size.shape}"
        )
    if np.any((legacy_type & NType.OUT_NODE) != 0):
        raise NotImplementedError(
            "from_legacy_triple: multi-output (OUT_NODE-tagged) trees are not "
            "implemented in the evoxc GP stack yet"
        )

    base = legacy_type & NType.TYPE_MASK
    known = (
        (base == NType.VAR)
        | (base == NType.CONST)
        | (base == NType.UFUNC)
        | (base == NType.BFUNC)
        | (base == NType.TFUNC)
    )
    if not bool(known.all()):
        bad = int(np.nonzero(~known)[0][0])
        raise ValueError(
            f"from_legacy_triple: node type {int(legacy_type.flat[bad])} of row "
            f"{bad // legacy_type.shape[1]} node {bad % legacy_type.shape[1]} is "
            "not a known NType marker"
        )

    leaf = base <= NType.CONST
    padding = legacy_size == 0
    function_opcode = legacy_value.astype(np.int32)
    if np.any((function_opcode < 0) | (function_opcode >= Func.END)):
        bad = int(np.nonzero(~leaf & ((function_opcode < 0) | (function_opcode >= Func.END)))[0][0])
        raise ValueError(
            f"from_legacy_triple: node {bad} carries the function id "
            f"{int(function_opcode.flat[bad])}, outside [0, {Func.END})"
        )

    opcode = np.where(padding, 0, np.where(leaf, base, function_opcode)).astype(
        np.int32
    )
    payload = np.where(padding, np.float32(0.0), np.where(leaf, legacy_value, np.float32(0.0)))
    return opcode, payload.astype(np.float32), legacy_size


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _resolve_config_pop(config, pop_size) -> Tuple[GPTreeConfig, int]:
    """``(tree config, pop_size)`` from a config or a ``(tree, gp)`` pair."""
    if isinstance(config, tuple):
        if len(config) != 2:
            raise ValueError(
                "a config pair must be (GPTreeConfig, GPConfig), got a "
                f"{len(config)}-tuple"
            )
        tree_config, gp_config = config
        if not isinstance(tree_config, GPTreeConfig):
            raise TypeError(
                "the first element of a config pair must be a GPTreeConfig, got "
                f"{type(tree_config).__name__}"
            )
        if not isinstance(gp_config, GPConfig):
            raise TypeError(
                "the second element of a config pair must be a GPConfig, got "
                f"{type(gp_config).__name__}"
            )
        if pop_size is None:
            pop_size = gp_config.pop_size
        elif pop_size != gp_config.pop_size:
            raise ValueError(
                f"pop_size={pop_size} contradicts the GPConfig's "
                f"pop_size={gp_config.pop_size}"
            )
        return tree_config, pop_size

    if not isinstance(config, GPTreeConfig):
        raise TypeError(
            "config must be a GPTreeConfig or a (GPTreeConfig, GPConfig) pair, "
            f"got {type(config).__name__}"
        )
    if pop_size is None:
        raise ValueError(
            "pop_size is required with a bare GPTreeConfig (pass a "
            "(GPTreeConfig, GPConfig) pair to take it from the GPConfig)"
        )
    return config, pop_size


def _padded_offsets(config: GPTreeConfig) -> Tuple[int, ...]:
    offsets = []
    running = 0
    for width in config.padded_widths():
        offsets.append(running)
        running += width
    return tuple(offsets)


def _slot_leaf_probs(config: GPTreeConfig) -> np.ndarray:
    """Leaf probability of every padded slot, by the depth of the slot."""
    values = [
        config.depth2leaf_prob_at(depth)
        for depth, width in enumerate(config.padded_widths())
        for _ in range(width)
    ]
    return np.asarray(values, dtype=np.float32)


def _generator_fn(config: GPTreeConfig, pop_size: int):
    """The traced generation graph of one ``(config, pop_size)`` (a closure).

    The closure keeps the config alive: ``evoxc.build_cached`` memoizes on the
    callable's identity, so the same object must be presented again (the
    executable memo in :func:`init_population` does that).
    """
    evoxc = _evoxc()
    # Imported here, not at module scope: `_runtime` imports this module (it is
    # the shared trace-primitive home of the operator layer).
    from ._runtime import columns as column_index, cumsum_axis, f32, i32

    widths = config.padded_widths()
    offsets = _padded_offsets(config)
    levels = config.levels
    total = sum(widths)
    arity_max = config.max_arity
    max_nodes = config.max_nodes
    columns = max_nodes + 1  # one scratch column absorbs the dropped slots
    scratch = max_nodes
    flat = pop_size * total

    leaf_probs = _slot_leaf_probs(config)
    cdf = np.asarray(config.roulette_funcs, dtype=np.float32)
    arity_table = np.asarray(config.func_arity, dtype=np.int32)
    constants = np.asarray(config.const_samples, dtype=np.float32)
    rank_pattern = np.arange(arity_max, dtype=np.int32)
    row_offsets = (np.arange(pop_size, dtype=np.int32) * columns).reshape(pop_size, 1)
    const_prob = float(config.const_prob)
    input_len = int(config.input_len)

    def generate(key):
        (
            k_leaf,
            k_func,
            k_const,
            k_const_index,
            k_var_index,
            k_next,
        ) = evoxc.random.split_n(key, 6)

        shape = (pop_size, total)
        scalar0f = evoxc.constant(evoxc.zeros((1,), evoxc.float32))
        scalar0i = evoxc.constant(evoxc.zeros((1,), evoxc.int32))
        # Typed rank-0 literals: a Python int/float would promote the whole
        # expression to int64/float64, which the strict stablehlo emitter
        # rejects at the first concatenate of mismatched element types.
        zero_i = i32(evoxc, 0)
        one_i = i32(evoxc, 1)
        zero_f = f32(evoxc, 0.0)

        # ---- node decisions of the whole padded population, in one batch ---
        r_leaf = evoxc.random.uniform(k_leaf, shape, 0.0, 1.0)
        is_func = evoxc.greater_equal(
            r_leaf, evoxc.constant(evoxc.from_numpy(leaf_probs))
        )

        r_func = evoxc.random.uniform(k_func, shape, 0.0, 1.0)
        cdf_constant = evoxc.constant(evoxc.from_numpy(cdf))
        # searchsorted(cdf, r, "right") == the number of cdf entries <= r, which
        # evoxc can only express as a sum of comparisons (no searchsorted).
        # Accumulated one entry at a time instead of as a 3-D reduce so the
        # temporary is one [pop, slots] tensor rather than [pop, slots, 29].
        count = evoxc.cast(evoxc.less_equal(cdf_constant[0], r_func), evoxc.int32)
        for index in range(1, Func.END):
            count = evoxc.add(
                count,
                evoxc.cast(evoxc.less_equal(cdf_constant[index], r_func), evoxc.int32),
            )
        func_id = evoxc.minimum(count, i32(evoxc, Func.END - 1))
        arity = evoxc.select(
            is_func,
            evoxc.gather(evoxc.constant(evoxc.from_numpy(arity_table)), func_id, axis=0),
            scalar0i[0],
        )

        r_const = evoxc.random.uniform(k_const, shape, 0.0, 1.0)
        is_const = evoxc.less_equal(r_const, const_prob)
        const_index = evoxc.random.randint(
            k_const_index, shape, 0, int(constants.size), dtype=evoxc.int32
        )
        const_value = evoxc.gather(
            evoxc.constant(evoxc.from_numpy(constants)), const_index, axis=0
        )
        var_index = evoxc.random.randint(
            k_var_index, shape, 0, input_len, dtype=evoxc.int32
        )
        leaf_payload = evoxc.select(
            is_const, const_value, evoxc.cast(var_index, evoxc.float32)
        )
        payload = evoxc.select(is_func, zero_f, leaf_payload)
        opcode = evoxc.select(
            is_func, func_id, evoxc.cast(is_const, evoxc.int32)
        )

        # ---- which padded slots hold a real node --------------------------
        # A slot exists iff its parent is real and its rank is inside the
        # parent's arity; the root always exists.
        real_levels = [evoxc.constant(evoxc.ones((pop_size, 1), evoxc.bool_))]
        arity_levels = [arity[:, :1]]
        rank3 = evoxc.reshape(evoxc.constant(evoxc.from_numpy(rank_pattern)), (1, 1, arity_max))
        for depth in range(1, levels):
            width = widths[depth]
            rank_ok = evoxc.less(
                rank3,
                evoxc.reshape(arity_levels[-1], (pop_size, widths[depth - 1], 1)),
            )
            real = evoxc.reshape(
                evoxc.logical_and(
                    evoxc.reshape(real_levels[-1], (pop_size, widths[depth - 1], 1)),
                    rank_ok,
                ),
                (pop_size, width),
            )
            lo, hi = offsets[depth], offsets[depth] + width
            real_levels.append(real)
            arity_levels.append(
                evoxc.multiply(arity[:, lo:hi], evoxc.cast(real, evoxc.int32))
            )

        # ---- bottom-up subtree sizes (children of a slot are consecutive) --
        sizes = [None] * levels
        sizes[-1] = evoxc.cast(real_levels[-1], evoxc.int32)
        for depth in range(levels - 2, -1, -1):
            children = evoxc.reshape(
                sizes[depth + 1], (pop_size, widths[depth], arity_max)
            )
            # `sum` accumulates in 64-bit, so the cast back keeps `size` int32
            # (a mixed-dtype `concatenate` is rejected by the strict emitter).
            sizes[depth] = evoxc.multiply(
                evoxc.add(evoxc.cast(evoxc.sum(children, axes=2), evoxc.int32), one_i),
                evoxc.cast(real_levels[depth], evoxc.int32),
            )

        # ---- top-down pre-order indices -----------------------------------
        # pre(child of rank r) = pre(parent) + 1 + sizes of the smaller ranks.
        prefixes = [evoxc.constant(evoxc.zeros((pop_size, 1), evoxc.int32))]
        for depth in range(1, levels):
            siblings = evoxc.reshape(
                sizes[depth], (pop_size, widths[depth - 1], arity_max)
            )
            smaller = evoxc.subtract(
                cumsum_axis(evoxc, siblings, 2, zero_i), siblings
            )
            prefix = evoxc.add(
                evoxc.reshape(prefixes[-1], (pop_size, widths[depth - 1], 1)),
                evoxc.add(one_i, smaller),
            )
            prefixes.append(
                evoxc.select(
                    real_levels[depth],
                    evoxc.reshape(prefix, (pop_size, widths[depth])),
                    i32(evoxc, scratch),
                )
            )

        # ---- the padded grid back into pre-order --------------------------
        # The layout is flattened to 1-D because evoxc.scatter takes the
        # indices' shape as the "middle" axes and does not broadcast them
        # against the leading axis of a 2-D base; every row owns a disjoint
        # slice of the flat buffer, so this is the same write.
        #
        # The three channels are NOT three scatters: a program with two
        # `stablehlo.scatter` ops whose window is scalar (rank-1 operand) fails
        # to compile on the iree backend ("unresolved materialization from
        # tensor<Kx1xi32> to tensor<Kxi32>"), so ONE scatter builds the inverse
        # map (destination column -> source slot) once and each channel is then
        # one row-wise gather.  Duplicate destinations (every dropped slot lands
        # on the scratch column) are masked out below, so last-write-wins on
        # them is irrelevant.
        index_flat = evoxc.reshape(
            evoxc.add(
                evoxc.concatenate(prefixes, axis=1),
                evoxc.constant(evoxc.from_numpy(row_offsets)),
            ),
            (flat,),
        )
        slot_table = np.arange(flat, dtype=np.int32)
        inverse_flat = evoxc.scatter(
            evoxc.broadcast(zero_i, (pop_size * columns,)),
            index_flat,
            evoxc.constant(evoxc.from_numpy(slot_table)),
            axis=0,
        )
        # The root's subtree size IS the row's live node count, so this is the
        # padding mask without a second reduction.
        keep = evoxc.less(column_index(evoxc, columns), sizes[0])
        pad_len = pop_size * columns - flat
        outputs = []
        for values in (opcode, payload, evoxc.concatenate(sizes, axis=1)):
            zero = (
                zero_f if values.dtype == evoxc.float32 else zero_i
            )
            flat_values = evoxc.reshape(values, (flat,))
            if pad_len > 0:
                flat_values = evoxc.concatenate(
                    (flat_values, evoxc.broadcast(zero, (pad_len,))), axis=0
                )
            pulled = evoxc.reshape(
                evoxc.gather(flat_values, inverse_flat, axis=0),
                (pop_size, columns),
            )
            buffer = evoxc.select(keep, pulled, zero)
            outputs.append(
                evoxc.slice(
                    buffer,
                    start=(0, 0),
                    lengths=(pop_size, max_nodes),
                )
            )

        return outputs[0], outputs[1], outputs[2], k_next

    return generate


def _resolve_key(evoxc, key):
    """A concrete ``evoxc`` key from an int seed or an explicit key tensor."""
    if isinstance(key, bool):
        raise TypeError("key must be an int seed or an evoxc key tensor, got bool")
    if isinstance(key, int):
        return evoxc.random.key(key)
    if isinstance(key, evoxc.Tensor):
        return key
    raise TypeError(
        "key must be an int seed or a concrete evoxc key (evoxc.random.key(...)), "
        f"got {type(key).__name__}"
    )


def _build(tree_config: GPTreeConfig, pop_size: int, backend: str, device: str, key):
    """The memoized executable of one ``(config, pop_size, backend, device, key)``."""
    evoxc = _evoxc()
    memo_key = (
        tree_config,
        pop_size,
        backend,
        device,
        tuple(int(size) for size in key.shape),
        str(key.dtype),
    )
    entry = _EXECUTABLE_MEMO.get(memo_key)
    if entry is not None:
        return entry[1]
    generate = _generator_fn(tree_config, pop_size)
    executable = evoxc.build_cached(
        generate,
        evoxc.TensorSpec(tuple(int(size) for size in key.shape), key.dtype),
        backend=backend,
        device=device,
        cache=_compile_cache(),
    )
    _EXECUTABLE_MEMO[memo_key] = (generate, executable)
    return executable


def init_population(
    config,
    key,
    pop_size: Optional[int] = None,
    backend: str = "numpy",
    device: str = "cpu",
    validate: bool = True,
) -> GPState:
    """Draw a random population, one traced graph per ``(config, pop_size)``.

    The decision structure is the tree layer's (``../tree/generation.py``, see
    ``DESIGN.md``): a node at depth ``d`` is a function iff
    ``r >= depth2leaf_probs[d]``, its id is the number of cdf entries ``<= r``,
    its arity follows from the id; a leaf is a constant with probability
    ``const_prob`` (value drawn uniformly from the constant table) and a
    variable index ``% input_len`` otherwise.  The population is grown on the
    padded complete-tree layout and scattered once into pre-order, so the whole
    generation is a single `evoxc.run` of a static-shape graph.

    Args:
        config: a :class:`~evogp.gp.config.GPTreeConfig` (then ``pop_size`` is
            required) or a ``(GPTreeConfig, GPConfig)`` pair (then ``pop_size``
            defaults to ``GPConfig.pop_size``).
        key: an int seed (``evoxc.random.key(seed)``) or a concrete
            ``evoxc`` key tensor.
        pop_size: number of trees; see ``config``.
        backend: evoxc backend name (``"numpy"`` default; ``"iree"`` for a
            device box -- the numpy reference backend is the only one that
            implements ``evoxc.random`` in this evoxc revision).
        device: evoxc device string.
        validate: run :func:`validate_channels` on the result (default on; it
            is a host-side pass over the population, not a hot path).

    Returns:
        A :class:`~evogp.gp.state.GPState` with the three channels, an
        all-``-inf`` (unevaluated) fitness vector, and ``key`` set to the next
        generation's key.

    Raises:
        NotImplementedError: for ``output_len > 1`` (multi-output out-tagging is
            not implemented yet).
    """
    tree_config, pop_size = _resolve_config_pop(config, pop_size)
    if tree_config.output_len != 1:
        raise NotImplementedError(
            f"init_population: output_len={tree_config.output_len} > 1 needs the "
            "multi-output out-tagging (out_prob / NType.OUT_NODE) that the "
            "evoxc GP stack does not implement yet; use output_len=1"
        )
    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError(f"pop_size must be a positive integer, got {pop_size!r}")

    evoxc = _evoxc()
    # A function-level import: `_runtime` imports this module at module scope.
    from . import _runtime

    key_tensor = _runtime.to_device(_resolve_key(evoxc, key), device)
    executable = _build(tree_config, pop_size, backend, device, key_tensor)
    opcode, payload, size, next_key = evoxc.run(executable, key_tensor)

    state = GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=_runtime.unevaluated(pop_size, device=device),
        key=next_key,
    )
    if validate:
        validate_channels(state, input_len=tree_config.input_len)
    return state
