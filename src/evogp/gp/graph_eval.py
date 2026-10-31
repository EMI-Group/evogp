"""Device-resident in-graph evaluation of a whole population's SR fitness.

:mod:`evogp.gp.problem` already evaluates a population inside ONE traced
``evoxc`` graph, but it does so from the **host**: ``_graph_inputs`` turns the
three channel tensors into the seven structure arrays the traced body consumes
(``kind``, ``func_id``, ``const_value``, ``var_index``, ``child1``, ``child2``,
``child3``), so every call pays a host pre-pass over ``pop_size * max_nodes`` and
uploads seven arrays.  That pre-pass is 0.15 / 0.29 / 1.12 ms at pop 64 / 256 /
1000 -- i.e. the same order as the graph itself -- and it forces the channel
tensors back to the host even when the previous generation produced them on the
device.

This module computes that pre-pass **inside the graph**: the three channels are
graph inputs and the seven structure arrays are traced from them, so a
generation can be ``variation.run -> eval.run`` with the channels never leaving
the device (no ``np.asarray``, no ``.numpy()``, no per-call upload).  The traced
body itself is *not* duplicated: :func:`sr_fitness_channels` builds the prelude
and then reuses :func:`evogp.gp.problem._graph_sr_fitness` (the semantics of
record, shared with :func:`evogp.gp.problem.sr_fitness_graph`), so the two
in-graph routes cannot drift apart numerically.

Public surface:

* :func:`sr_fitness_channels` -- ``(pop_size,)`` float32 MAXIMIZED fitness as an
  ``evoxc`` tensor **on the device** (no readback);
* :func:`sr_fitness_channels_numpy` -- the same call with the one explicit
  device-to-host readback;
* :func:`readback_fitness` -- the readback helper for a fitness tensor the
  caller already holds;
* :class:`GraphEvaluator` -- the caller-facing object: it owns ``x``/``y`` (so
  they are uploaded once, at construction) and exposes
  ``evaluate(channels) -> evoxc.Tensor``, the plug-in surface of
  ``DESIGN.md`` section "Evaluation and the driver";
* :func:`device_graph_stats` / :func:`clear_device_graph_cache` -- the memo
  counters of this layer.

Three constraints shape the code:

**Numerics are shared, structure is new.**  The prelude reproduces
``problem._graph_inputs`` op for op (``function = size > 1``, ``kind`` =
``FUNC_TYPE[opcode]`` for a function node else ``opcode``, ...,
``c1 = i + 1`` / ``c2 = c1 + size[c1]`` / ``c3 = c2 + size[c2]`` clipped into the
row), so the new route and the host route agree bit for bit on the numpy
backend.  The prelude is computed in the ``[pop_size, width]`` channel layout
(the layout ``pop_size``/``width`` are read in) and transposed to the
``[width, pop_size]`` layout ``_graph_body`` slices; the row-wise child lookups
go through :func:`evogp.gp._runtime.take_per_row`, because ``evoxc.gather`` is
numpy ``take``, not ``take_along_axis``.

**The used function ids are static.**  The traced dispatch is a select chain
over the population's used function ids (``problem._graph_sr_fitness``), and the
set cannot be read off the channels without a host sync -- so it becomes a
static argument.  The default is *every* id of ``Func`` (always correct, and the
mask below is then not traced at all), and passing the tree config (or its
``using_funcs``) narrows the chain to the ids a population of that config can
contain, which is what a real run should do.  Because a too-narrow set cannot
raise in-graph, it is caught structurally: a row carrying a function id outside
the set gets ``-inf`` fitness instead of the chain's silently-wrong ``0.0``
(:func:`_uncovered_rows`).

**No host validation, no host sanitizing.**  The device path cannot run
``encoding.validate_channels`` (it is a Python loop over a numpy view of the
channels, i.e. a readback); well-formedness is the operator layer's contract,
exactly as it is for the offspring the driver does not re-validate.  The
non-finite -> ``-inf`` mapping of ``problem._fitness_from_loss`` IS applied, in
graph (``isnan`` plus ``|fitness| == inf``), so the returned device vector has
the repo-wide fitness contract without a readback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import _runtime
from ._constants import FUNC_IDS, FUNC_TYPE, Func, NType
from .config import GPTreeConfig
from .problem import _graph_sr_fitness

__all__ = [
    "ALL_FUNC_IDS",
    "resolve_used_ids",
    "sr_fitness_channels",
    "sr_fitness_channels_numpy",
    "readback_fitness",
    "GraphEvaluator",
    "device_graph_stats",
    "clear_device_graph_cache",
]

#: Every function id of :class:`evogp.gp._constants.Func` (the default
#: ``used_ids``: correct for any population, the widest dispatch chain).
ALL_FUNC_IDS: Tuple[int, ...] = tuple(range(Func.END))

#: ``(width, pop_size, n_rows, input_len, used_ids, use_mse, sanitize,
#: backend, device)`` -> ``(graph_fn, executable)``.
_DEVICE_MEMO: dict = {}


# --------------------------------------------------------------------------- #
# static arguments
# --------------------------------------------------------------------------- #
def resolve_used_ids(used_ids=None) -> Tuple[int, ...]:
    """The canonical ``(id, ...)`` tuple the traced dispatch chain covers.

    Args:
        used_ids: ``None`` for every id of :class:`Func` (:data:`ALL_FUNC_IDS`),
            a :class:`~evogp.gp.config.GPTreeConfig` (its ``using_funcs`` -- the
            narrow, fast and always-sufficient set for that config), or an
            iterable of ids / function names.

    Returns:
        A sorted, duplicate-free tuple of ids.

    Raises:
        TypeError: for a bare string (a function set must be a sequence).
        ValueError: for an empty set, an unknown name or an id outside
            ``[0, Func.END)``.

    Notes:
        The set must be a *superset* of the ids that can appear in the
        population: the dispatch chain is a select over exactly these ids.  A
        missing id cannot raise in-graph, so it is caught structurally instead --
        a row carrying a function id outside the set is marked ``-inf``
        ("unevaluable") rather than silently evaluated with that node's value
        left at ``0.0`` (:func:`_uncovered_rows`).  Widening the set is always
        safe; it only adds ops to the chain.
    """
    if used_ids is None:
        return ALL_FUNC_IDS
    if isinstance(used_ids, GPTreeConfig):
        used_ids = [name for name, _ in used_ids.using_funcs]
    if isinstance(used_ids, str):
        raise TypeError(
            "used_ids must be a sequence of ids/names or a GPTreeConfig, not a "
            f"bare string ({used_ids!r}); a single function is spelled "
            f"({used_ids!r},)"
        )
    ids = set()
    for item in used_ids:
        if isinstance(item, str):
            if item not in FUNC_IDS:
                raise ValueError(
                    f"unknown function name {item!r} in used_ids; the function "
                    f"set is {tuple(FUNC_IDS)}"
                )
            ids.add(FUNC_IDS[item])
            continue
        value = int(item)
        if not 0 <= value < Func.END:
            raise ValueError(
                f"used_ids entry {value} is outside [0, Func.END={Func.END})"
            )
        ids.add(value)
    if not ids:
        raise ValueError("used_ids must not be empty")
    return tuple(sorted(ids))


def _cfg(width, pop_size, n_rows, input_len, used_ids, use_mse, sanitize) -> tuple:
    """The ONE static tuple the traced entry point closes over.

    The first six fields are exactly ``problem``'s ``cfg`` (so
    ``problem._graph_sr_fitness`` can be called with ``cfg[:6]``); the seventh
    switches the in-graph non-finite mapping.  Build and run share this builder,
    so the executable's static arguments and the ones ``evoxc.run`` validates
    cannot drift apart.
    """
    return (
        int(width),
        int(pop_size),
        int(n_rows),
        int(input_len),
        tuple(used_ids),
        bool(use_mse),
        bool(sanitize),
    )


# --------------------------------------------------------------------------- #
# the traced prelude: the seven structure arrays, computed in-graph
# --------------------------------------------------------------------------- #
def _func_type_table(evoxc_module):
    """``FUNC_TYPE`` as a ``[Func.END]`` int32 graph constant (the lookup table)."""
    table = np.asarray(FUNC_TYPE, dtype=np.int32)
    return evoxc_module.constant(evoxc_module.from_numpy(table))


def _uncovered_rows(evoxc_module, function, kind, func_id, used_ids):
    """``[pop_size]`` bool: rows carrying a function id outside ``used_ids``.

    The traced dispatch is a select chain over ``used_ids``, so a function node
    whose id is not in the chain contributes the chain's initial ``0.0`` instead
    of its own value -- a silently WRONG fitness rather than an error.  This
    mask turns that into the repo-wide "unevaluable" marker (``-inf``), which is
    the only signal a graph can produce without a host sync.

    A ``TFUNC`` node is excluded when ``Func.IF`` is in ``used_ids``: ternary
    ``if`` is dispatched structurally (``op_i == NType.TFUNC``), so it needs no
    id in the chain.  Without ``Func.IF`` in the set the body does not emit the
    ternary at all (``problem._graph_body``: its third-child gather is the
    dearest read of the walk and a set without ``if`` cannot produce one), so
    such a row IS uncovered and is marked ``-inf`` like any other id outside the
    chain.  With the default ``used_ids`` (every id of :class:`Func`) the mask
    cannot fire at all, which is why :func:`_structure_inputs` skips tracing it
    in that case.
    """
    if len(used_ids) >= Func.END:  # pragma: no cover - the caller skips it
        return None
    covered = evoxc_module.equal(func_id, _runtime.i32(evoxc_module, used_ids[0]))
    for fid in used_ids[1:]:
        covered = evoxc_module.logical_or(
            covered, evoxc_module.equal(func_id, _runtime.i32(evoxc_module, fid))
        )
    if Func.IF in used_ids:
        dispatchable = function
    else:
        dispatchable = evoxc_module.logical_and(
            function,
            evoxc_module.not_equal(kind, _runtime.i32(evoxc_module, NType.TFUNC)),
        )
    unknown = evoxc_module.logical_and(
        dispatchable, evoxc_module.logical_not(covered)
    )
    return evoxc_module.greater(
        evoxc_module.sum(evoxc_module.cast(unknown, evoxc_module.int32), axes=1),
        _runtime.i32(evoxc_module, 0),
    )


def _structure_inputs(evoxc_module, opcode, payload, size, cfg) -> tuple:
    """Trace the seven ``[width, pop_size]`` structure arrays from the channels.

    Op-for-op counterpart of the host pre-pass ``problem._graph_inputs``:

    * ``function = size > 1`` (a leaf is exactly a node with ``size == 1``);
    * ``kind = FUNC_TYPE[clip(opcode, 0, END-1)]`` on a function node, else the
      ``NType`` leaf marker;
    * ``func_id = opcode`` on a function node, else ``0``;
    * ``const_value = payload`` on a ``CONST`` leaf, else ``0.0`` (a function
      node's payload is always ``0.0`` by contract, the selects keep the
      invariant explicit);
    * ``var_index = clip(trunc(payload), 0, input_len - 1)`` on a ``VAR`` leaf,
      else ``0``;
    * ``c1 = i + 1``, ``c2 = c1 + size[c1]``, ``c3 = c2 + size[c2]``, each
      clipped into the row.

    Args:
        evoxc_module: the traced ``evoxc`` module.
        opcode: int32 ``[pop_size, width]`` function ids / ``NType`` markers.
        payload: float32 ``[pop_size, width]`` constant value / variable index.
        size: int32 ``[pop_size, width]`` subtree sizes.
        cfg: the static tuple of :func:`_cfg`.

    Returns:
        ``(structure, uncovered)`` where ``structure`` is
        ``(kind, func_id, const_value, var_index, child1, child2, child3)`` --
        all ``[width, pop_size]`` (the layout ``problem._graph_body`` slices),
        int32 dtypes except ``const_value`` (float32) -- and ``uncovered`` is the
        ``[pop_size]`` bool of :func:`_uncovered_rows`, or ``None`` when
        ``used_ids`` covers every id of :class:`Func` (nothing can be
        uncovered then, so the mask is not traced).

    Notes:
        * Every literal goes through :func:`_runtime.i32` / :func:`_runtime.f32`:
          a bare Python scalar would promote the arithmetic to int64/float64,
          which the ``iree`` backend rejects and which would also change the
          ``kind``/``func_id`` dtypes the body compares against.
        * The variable index clamps in FLOAT before the int cast, which is
          value-equal to the host's ``trunc`` then ``clip`` on int64 for every
          payload (``trunc(clip(p, 0, L-1)) == clip(trunc(p), 0, L-1)``) and
          additionally cannot overflow the int32 cast -- the host pre-pass needs
          an int64 intermediate for that reason.
        * ``c2``/``c3`` use the host formula verbatim (``i + 1 + s``, not
          ``c1 + s``); the two agree because the only column where ``c1`` is
          already clipped is the last one, where both forms clip to ``width-1``.
    """
    width = int(cfg[0])
    pop_size = int(cfg[1])
    input_len = int(cfg[3])

    zero_i = _runtime.i32(evoxc_module, 0)
    one_i = _runtime.i32(evoxc_module, 1)
    last_id = _runtime.i32(evoxc_module, Func.END - 1)
    last_column = _runtime.i32(evoxc_module, width - 1)
    zero_f = _runtime.f32(evoxc_module, 0.0)

    function = evoxc_module.greater(size, one_i)
    is_const = evoxc_module.equal(opcode, _runtime.i32(evoxc_module, NType.CONST))

    kind = evoxc_module.select(
        function,
        evoxc_module.gather(
            _func_type_table(evoxc_module),
            evoxc_module.clamp(opcode, zero_i, last_id),
            axis=0,
        ),
        opcode,
    )
    func_id = evoxc_module.select(function, opcode, zero_i)
    const_value = evoxc_module.select(
        function, zero_f, evoxc_module.select(is_const, payload, zero_f)
    )
    var_index = evoxc_module.select(
        evoxc_module.logical_or(function, is_const),
        zero_i,
        evoxc_module.cast(
            evoxc_module.clamp(
                payload, zero_f, _runtime.f32(evoxc_module, float(input_len - 1))
            ),
            evoxc_module.int32,
        ),
    )

    columns = _runtime.columns(evoxc_module, width)  # [1, width] int32
    child1 = evoxc_module.broadcast(
        evoxc_module.minimum(evoxc_module.add(columns, one_i), last_column),
        (pop_size, width),
    )
    size1 = _runtime.take_per_row(evoxc_module, size, child1)
    child2 = evoxc_module.minimum(
        evoxc_module.add(evoxc_module.add(columns, one_i), size1), last_column
    )
    size2 = _runtime.take_per_row(evoxc_module, size, child2)
    child3 = evoxc_module.minimum(evoxc_module.add(child2, size2), last_column)

    structure = tuple(
        evoxc_module.transpose(tensor, (1, 0))
        for tensor in (
            kind,
            func_id,
            const_value,
            var_index,
            child1,
            child2,
            child3,
        )
    )
    used_ids = tuple(int(value) for value in cfg[4])
    uncovered = _uncovered_rows(evoxc_module, function, kind, func_id, used_ids)
    return structure, uncovered


def _sanitize_fitness(evoxc_module, fitness):
    """``fitness`` with every non-finite entry replaced by ``-inf``, in graph.

    ``problem._fitness_from_loss`` does this on the host (``~np.isfinite`` ->
    ``-inf``); the device route must not read back to do it, so it is composed
    from ``isnan`` (``not_equal(x, x)``) and ``|x| == inf`` -- exact for float32,
    since no finite float32 equals infinity.
    """
    infinity = _runtime.f32(evoxc_module, float("inf"))
    not_finite = evoxc_module.logical_or(
        evoxc_module.isnan(fitness),
        evoxc_module.equal(evoxc_module.abs(fitness), infinity),
    )
    return evoxc_module.select(
        not_finite, _runtime.f32(evoxc_module, float("-inf")), fitness
    )


def _device_sr_fitness(opcode, payload, size, x, y, cfg):
    """Traced entry point of :func:`sr_fitness_channels` (module-level identity).

    Signature mirrors ``problem._graph_sr_fitness`` (three int32/float32 channel
    specs, the data, the targets, the static ``cfg``) so ``evoxc.build_cached``
    memoizes one program per static configuration; the three extra steps are the
    prelude, the optional in-graph non-finite mapping and the ``-inf`` mark of a
    row carrying a function id outside ``used_ids``.
    """
    import evoxc  # noqa: PLC0415 - the trace-time module (never at import time)

    structure, uncovered = _structure_inputs(evoxc, opcode, payload, size, cfg)
    fitness = _graph_sr_fitness(*structure, x, y, cfg[:6])
    if bool(cfg[6]):
        fitness = _sanitize_fitness(evoxc, fitness)
    if uncovered is not None:
        fitness = evoxc.select(
            uncovered, _runtime.f32(evoxc, float("-inf")), fitness
        )
    return fitness


# --------------------------------------------------------------------------- #
# the memoized executable
# --------------------------------------------------------------------------- #
def _device_key(device) -> str:
    """A canonical ``"kind:index"`` spelling of a device argument."""
    resolved = _device_object(device)
    return f"{resolved.kind}:{resolved.index}"


def _executable(
    width, pop_size, n_rows, input_len, used_ids, use_mse, sanitize, backend, device
):
    """The memoized executable of one ``(shape, used ids, knobs, backend)``."""
    cfg = _cfg(width, pop_size, n_rows, input_len, used_ids, use_mse, sanitize)
    memo_key = (cfg, str(backend), _device_key(device))
    evoxc_module = _runtime.evoxc()
    column = evoxc_module.TensorSpec((int(pop_size), int(width)), evoxc_module.int32)
    payload = evoxc_module.TensorSpec(
        (int(pop_size), int(width)), evoxc_module.float32
    )
    rows = evoxc_module.TensorSpec((int(n_rows), int(input_len)), evoxc_module.float32)
    targets = evoxc_module.TensorSpec((int(n_rows),), evoxc_module.float32)
    specs = (column, payload, column, rows, targets, cfg)
    return _runtime.executable(
        _DEVICE_MEMO,
        memo_key,
        lambda: _device_sr_fitness,
        specs,
        backend,
        _device_object(device),
    )


def device_graph_stats() -> dict:
    """Counters of this layer (its graph memo + the shared ``CompileCache``)."""
    from . import encoding

    stats = _runtime.stats(_DEVICE_MEMO)
    stats["compile_cache_maxsize"] = encoding._cache_size()  # noqa: SLF001
    return stats


def clear_device_graph_cache() -> None:
    """Drop every memoized device graph (the shared compile cache is untouched).

    ``evogp.gp.encoding.clear_cache()`` owns the ``CompileCache``; the operator
    modules reset their own memos the same way (:func:`_runtime.reset`).
    """
    _runtime.reset(_DEVICE_MEMO)


# --------------------------------------------------------------------------- #
# argument preparation (the only place a host array is touched)
# --------------------------------------------------------------------------- #
def _device_object(device):
    """``device`` as an :class:`evoxc.Device`.

    Accepts an ``evoxc.Device``, a kind string (``"cpu"``, ``"rocm"``) or a
    ``"kind:index"`` string.  The last spelling is split HERE on purpose:
    ``evoxc`` normalizes a *string* device as a bare kind at the build boundary
    (only ``EVOXC_DEVICE`` is parsed as ``"kind[:index]"``), so ``"rocm:0"``
    passed straight through would reach IREE as the nonsense kind ``"rocm:0"``.
    """
    evoxc_module = _runtime.evoxc()
    if isinstance(device, evoxc_module.Device):
        return device
    if isinstance(device, str):
        kind, sep, index = device.partition(":")
        kind = kind.strip()
        if not kind or (sep and not index.strip()):
            raise ValueError(
                f"device must be a kind string or 'kind:index', got {device!r}"
            )
        return evoxc_module.Device(kind, int(index) if sep else 0)
    raise TypeError(
        "device must be a kind string (e.g. 'cpu', 'rocm:0') or an "
        f"evoxc.Device, got {type(device).__name__}"
    )


def _channel_triple(channels):
    """``(opcode, payload, size, pop_size, width)`` -- no host conversion.

    A :class:`~evogp.gp.state.GPState` or the ``(opcode, payload, size)`` triple
    of ``evoxc`` tensors (host arrays are accepted on the numpy backend, where
    ``evoxc.run`` takes them as leaves).  Unlike the host paths
    (``encoding._channel_arrays``), a single individual's three 1-D channel rows
    are NOT promoted here: promoting needs ``evoxc.reshape``, which only exists
    inside a trace, so the device path requires the 2-D population form.
    """
    opcode, payload, size = _runtime.as_channels(channels)
    shape = tuple(int(value) for value in opcode.shape)
    if len(shape) != 2:
        raise ValueError(
            "the device evaluation path needs 2-D [pop_size, max_nodes] "
            f"channels, got shape {shape}; a single individual must be "
            "reshaped to a population of one inside a trace (or evaluated "
            "through evogp.gp.problem, which accepts the 1-D form)"
        )
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError(f"channels must be non-empty, got shape {shape}")
    for name, leaf in (("payload", payload), ("size", size)):
        if tuple(int(value) for value in leaf.shape) != shape:
            raise ValueError(
                "the three channels must share one shape, got opcode "
                f"{shape} and {name} {tuple(int(v) for v in leaf.shape)}"
            )
    return opcode, payload, size, shape[0], shape[1]


def _data_tensor(value, name: str, device):
    """``value`` as an ``evoxc`` tensor on ``device`` -- an upload only if needed.

    An ``evoxc.Tensor`` passes through untouched (no copy, no device check: an
    evoxc executable rejects a wrong-device input at the run boundary, and
    evoxc never transfers implicitly).  A host array is wrapped once
    (``from_numpy`` is zero-copy) and moved to ``device`` -- the convenience
    path for a one-off call; :class:`GraphEvaluator` does it once at
    construction so the per-generation call converts nothing.
    """
    evoxc_module = _runtime.evoxc()
    if isinstance(value, evoxc_module.Tensor):
        return value
    array = np.asarray(value, dtype=np.float32)
    if np.isnan(array).any() or np.isposinf(array).any() or np.isneginf(array).any():
        raise ValueError(f"{name} must be finite, got non-finite entries")
    return evoxc_module.from_numpy(np.ascontiguousarray(array)).to(
        _device_object(device)
    )


def _prepare_data(x, y, input_len, device):
    """``(x, y, n_rows, input_len)`` as ``evoxc`` tensors on ``device``."""
    evoxc_module = _runtime.evoxc()
    if isinstance(x, evoxc_module.Tensor):
        if len(x.shape) != 2:
            raise ValueError(f"x must be 2-D [n_rows, input_len], got {x.shape}")
        data = x
    else:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 1:  # one data row, like problem._prepare_data
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"x must be 1-D or 2-D, got shape {array.shape}")
        data = _data_tensor(array, "x", device)
    n_rows, columns = (int(value) for value in data.shape)
    if n_rows == 0 or columns == 0:
        raise ValueError(f"x must be non-empty, got shape {data.shape}")
    if input_len is not None and int(input_len) != columns:
        raise ValueError(
            f"input_len={input_len} contradicts x's {columns} columns"
        )
    if isinstance(y, evoxc_module.Tensor):
        if len(y.shape) != 1:
            raise ValueError(f"y must be 1-D [n_rows], got {y.shape}")
        targets = y
    else:
        targets = _data_tensor(np.asarray(y, dtype=np.float32).reshape(-1), "y", device)
    if int(targets.shape[0]) != n_rows:
        raise ValueError(
            f"y has {int(targets.shape[0])} entries but x has {n_rows} rows"
        )
    return data, targets, n_rows, columns


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def sr_fitness_channels(
    channels,
    x,
    y,
    *,
    use_mse: bool = True,
    input_len: Optional[int] = None,
    used_ids=None,
    sanitize: bool = True,
    backend: str = "numpy",
    device="cpu",
):
    """``(pop_size,)`` float32 MAXIMIZED SR fitness, computed entirely in-graph.

    The device-resident counterpart of
    :func:`evogp.gp.problem.sr_fitness_graph`: the population's structure
    (which cell is a leaf, its function id, where its children are) is traced
    from the channel tensors, so no host pre-pass and no per-call upload
    happen.  The loss is reduced in-graph and the result is returned **as an
    ``evoxc`` tensor on ``device``** -- read it back with
    :func:`readback_fitness` (or call :func:`sr_fitness_channels_numpy`).

    Args:
        channels: a :class:`~evogp.gp.state.GPState` or the ``(opcode, payload,
            size)`` triple, 2-D ``[pop_size, max_nodes]``, int32/float32/int32.
            Device tensors pass through untouched.
        x: ``[n_rows, input_len]`` float32 -- an ``evoxc`` tensor on ``device``
            (the steady-state call), or a host array, which is uploaded once.
        y: ``[n_rows]`` float32 targets, same options as ``x``.
        use_mse: ``True`` for the squared error, ``False`` for the absolute one
            (``SymbolicRegression``'s ``use_mse``).
        input_len: when given it must match ``x``'s column count (the ``VAR``
            bound baked into the graph).
        used_ids: the ids the traced select chain covers (see
            :func:`resolve_used_ids`); ``None`` = every id of ``Func``, which is
            always correct (and skips the uncovered-id mask entirely) -- pass
            the tree config (or its ``using_funcs``) in a real run.  A population
            carrying an id outside the set is marked ``-inf``, never silently
            mis-evaluated.
        sanitize: map a non-finite fitness to ``-inf`` in-graph
            (``problem._fitness_from_loss``'s rule). ``False`` returns the raw
            negated loss, which is only useful for parity diagnostics.
        backend: evoxc backend name (``"numpy"`` default, ``"iree"`` device).
        device: evoxc device -- a kind string (``"cpu"``, ``"rocm"``,
            ``"rocm:0"``) or an :class:`evoxc.Device`.

    Returns:
        The ``(pop_size,)`` float32 fitness as an ``evoxc`` tensor on ``device``.

    Raises:
        ValueError: for a channel/data shape mismatch (structure is NOT
            validated -- that needs a readback; see the module docstring).
        RuntimeError: when ``evoxc`` is missing (the stack's usual error).

    Notes:
        The graph is memoized per ``(width, pop_size, n_rows, input_len,
        used_ids, use_mse, sanitize, backend, device)`` in this module's memo
        and the subtree's shared ``CompileCache``; on a device backend the first
        call of a shape compiles (seconds) and every later generation of the
        same shape hits.
    """
    opcode, payload, size, pop_size, width = _channel_triple(channels)
    data, targets, n_rows, columns = _prepare_data(x, y, input_len, device)
    used = resolve_used_ids(used_ids)
    executable = _executable(
        width, pop_size, n_rows, columns, used, use_mse, sanitize, backend, device
    )
    cfg = _cfg(width, pop_size, n_rows, columns, used, use_mse, sanitize)
    return _runtime.evoxc().run(executable, opcode, payload, size, data, targets, cfg)


def sr_fitness_channels_numpy(
    channels, x, y, *, backend: str = "numpy", device="cpu", **options
) -> np.ndarray:
    """:func:`sr_fitness_channels` with the one explicit device-to-host readback.

    Same arguments and semantics; the returned ``(pop_size,)`` numpy array is
    fresh and writable and maps a non-finite entry to ``-inf`` again (a no-op
    after the in-graph ``sanitize``, a guard for ``sanitize=False``).
    """
    fitness = sr_fitness_channels(
        channels, x, y, backend=backend, device=device, **options
    )
    return readback_fitness(fitness, int(fitness.shape[0]))


def readback_fitness(fitness, pop_size: Optional[int] = None) -> np.ndarray:
    """A fresh ``(pop_size,)`` float32 numpy copy of a fitness tensor.

    Moves a non-CPU tensor to the host explicitly (``evoxc`` has no implicit
    device-to-host transfer, so ``Tensor.numpy()`` on a device payload raises),
    then applies the repo-wide ``non-finite -> -inf`` rule.

    Args:
        fitness: an ``evoxc`` tensor or any array-like of fitness values.
        pop_size: when given it must match the flattened length.

    Returns:
        A writable float32 numpy array.
    """
    evoxc_module = _runtime.evoxc()
    if isinstance(fitness, evoxc_module.Tensor):
        host = fitness
        if host.device.kind != "cpu":
            host = host.to(evoxc_module.Device("cpu", 0))
        array = np.array(host.numpy(), dtype=np.float32, copy=True)
    else:
        array = np.array(fitness, dtype=np.float32, copy=True)
    array = array.reshape(-1)
    if pop_size is not None and array.shape[0] != int(pop_size):
        raise ValueError(
            f"the graph returned {array.shape[0]} fitness values for a "
            f"population of {pop_size}"
        )
    array[~np.isfinite(array)] = -np.inf
    return array


# --------------------------------------------------------------------------- #
# the caller-facing evaluator
# --------------------------------------------------------------------------- #
class GraphEvaluator:
    """The device-resident ``evaluate(channels) -> (pop_size,)`` plug-in.

    Owns ``x``/``y`` so they are uploaded once, at construction, and evaluates a
    whole population with ONE traced graph per static shape.  This is the
    object a device driver loop plugs in (``DESIGN.md`` section "Evaluation and
    the driver"): the two calls of a generation become
    ``variation.run`` and ``GraphEvaluator.evaluate``, with the channel tensors
    staying on the device and the fitness arriving as a device tensor.

    Args:
        x: ``[n_rows, input_len]`` float32, an ``evoxc`` tensor on ``device`` or
            a host array (uploaded here, once).
        y: ``[n_rows]`` float32 targets, same options as ``x``.
        use_mse: squared (default) vs absolute error.
        input_len: must match ``x``'s columns when given.
        used_ids: the ids the dispatch chain covers (see
            :func:`resolve_used_ids`); pass the run's
            :class:`~evogp.gp.config.GPTreeConfig` for the narrow chain.
        sanitize: in-graph ``non-finite -> -inf`` (default on).
        backend: evoxc backend name.
        device: evoxc device (kind string or :class:`evoxc.Device`).

    Example:
        >>> evaluator = GraphEvaluator(x, y, used_ids=config, backend="iree",
        ...                            device="rocm:0")
        >>> fitness = evaluator.evaluate(state)      # evoxc tensor, no sync
        >>> values = evaluator.host_fitness(fitness)  # numpy, one sync
    """

    def __init__(
        self,
        x,
        y,
        *,
        use_mse: bool = True,
        input_len: Optional[int] = None,
        used_ids=None,
        sanitize: bool = True,
        backend: str = "numpy",
        device="cpu",
    ):
        self.backend = str(backend)
        self.device = device
        self.use_mse = bool(use_mse)
        self.sanitize = bool(sanitize)
        self.used_ids = resolve_used_ids(used_ids)
        self._x, self._y, self._n_rows, self._input_len = _prepare_data(
            x, y, input_len, device
        )

    # ---------------------------------------------------------------- knobs
    @property
    def x(self):
        """The ``[n_rows, input_len]`` data tensor this evaluator holds."""
        return self._x

    @property
    def y(self):
        """The ``[n_rows]`` target tensor this evaluator holds."""
        return self._y

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def input_len(self) -> int:
        return self._input_len

    @property
    def problem_dim(self) -> int:
        """``input_len`` -- the ``GenerateDescriptor.input_len`` companion."""
        return self._input_len

    @property
    def solution_dim(self) -> int:
        """``1`` -- single-output SR, like :class:`SymbolicRegression`."""
        return 1

    # ------------------------------------------------------------ evaluation
    def evaluate(self, channels):
        """``(pop_size,)`` float32 fitness as an ``evoxc`` tensor on ``device``."""
        opcode, payload, size, pop_size, width = _channel_triple(channels)
        return self._run(opcode, payload, size, pop_size, width)

    def evaluate_numpy(self, channels) -> np.ndarray:
        """:meth:`evaluate` plus the one explicit readback (numpy in, numpy out)."""
        fitness = self.evaluate(channels)
        return readback_fitness(fitness, int(fitness.shape[0]))

    def host_fitness(self, fitness) -> np.ndarray:
        """The numpy readback of a fitness tensor this evaluator produced."""
        return readback_fitness(fitness, int(fitness.shape[0]))

    def warm(self, channels=None, *, shape=None) -> "GraphEvaluator":
        """Build (or hit) the executable of a shape WITHOUT running it.

        Args:
            channels: a population whose shape should be warmed.
            shape: ``(pop_size, max_nodes)`` instead of ``channels`` -- lets a
                caller warm the graph before any population exists.

        Returns:
            ``self``, so construction can be chained.
        """
        if shape is None:
            _, _, _, pop_size, width = _channel_triple(channels)
        else:
            pop_size, width = (int(value) for value in shape)
        _executable(
            width,
            pop_size,
            self._n_rows,
            self._input_len,
            self.used_ids,
            self.use_mse,
            self.sanitize,
            self.backend,
            self.device,
        )
        return self

    def __call__(self, channels):
        """``evaluate`` -- so an evaluator is directly usable as ``evaluate=``."""
        return self.evaluate(channels)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"GraphEvaluator(n_rows={self._n_rows}, input_len={self._input_len}, "
            f"use_mse={self.use_mse}, used_ids={self.used_ids!r}, "
            f"backend={self.backend!r}, device={self.device!r})"
        )

    # --------------------------------------------------------------- private
    def _run(self, opcode, payload, size, pop_size, width):
        executable = _executable(
            width,
            pop_size,
            self._n_rows,
            self._input_len,
            self.used_ids,
            self.use_mse,
            self.sanitize,
            self.backend,
            self.device,
        )
        cfg = _cfg(
            width,
            pop_size,
            self._n_rows,
            self._input_len,
            self.used_ids,
            self.use_mse,
            self.sanitize,
        )
        return _runtime.evoxc().run(
            executable, opcode, payload, size, self._x, self._y, cfg
        )
