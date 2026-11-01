"""Mutation operators of the evoxc GP stack: full-population masked applies.

``evoxc`` has no ``nonzero`` and no boolean masking, so the torch family's
"select the rows that mutate, then edit them" becomes a **masked apply**: each
operator draws its candidate position/values for *every* row and folds the
result back with ``select(mask, mutated, original)``.  Nothing is shape-variant,
so one graph serves the whole run.

Three operators (``MUTATION_KINDS``), mirroring the deleted torch stack's
``mutation/``:

``point`` (``SinglePointMutation``)
    One node per row is replaced by a node of the *same kind*: a leaf keeps its
    ``VAR``/``CONST`` identity and gets a fresh payload (variable index /
    constant from the config's table); a function node swaps its opcode for a
    **same-arity** function of the config's set.  Arity is therefore preserved
    by construction and the ``size`` channel stays valid with no recomputation --
    the one thing this operator must never break.
``subtree`` (``DefaultMutation``)
    One node per row has its subtree replaced by a freshly generated random tree
    drawn from the *same* generation graph as
    :func:`evogp.gp.encoding.init_population`
    (``encoding._generator_fn``), spliced in with ``donor_pos = 0`` by
    :func:`evogp.gp.crossover.splice`.  Rows whose result would exceed
    ``max_nodes`` are left untouched, exactly like the legacy operator.
``hoist`` (``HoistMutation``)
    The whole tree becomes the subtree rooted at one drawn node: a copy of
    ``[pos, pos + size[pos])`` shifted to column 0, the rest padding.  It can
    only shrink a tree, so it never overflows.

Point mutation's same-arity table
---------------------------------
The config's function set is grouped by arity at graph-build time (three
candidate blocks -- arity 1, 2, 3 -- concatenated into one int32 table plus an
``offset``/``count`` pair indexed by arity).  A row's candidate count is
``count[arity(node)]``; a function node with no same-arity alternative in the
set is left alone (``count == 0``), which is why the ``randint`` upper bound is
``max(count, 1)`` -- a zero-width range is a runtime error, not a no-op.
Candidates are drawn uniformly inside the class (the legacy operator weights
them by the config's roulette cdf; uniform is the documented simplification).

Public surface
--------------
``apply_mutation(evoxc_module, tree_config, channels, key, probability, kind)``
    The trace-time entry point (used by :mod:`evogp.gp.variation`).
``mutate(config, state, key, ...)``
    Host-facing: one memoized graph per ``(tree_config, kind, prob, pop_size,
    max_nodes, backend, device)``, returning the offspring population with an
    unevaluated (``-inf``) fitness vector.
``point_mutation`` / ``subtree_mutation`` / ``hoist_mutation``
    The three operators on their own (each its own host-facing call).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _runtime, encoding
from ._constants import FUNC_ARITY, FUNC_IDS, NType
from .config import GPTreeConfig, MUTATION_KINDS
from .crossover import splice
from .state import GPState

__all__ = [
    "MUTATION_KINDS",
    "apply_mutation",
    "mutate",
    "point_mutation",
    "subtree_mutation",
    "hoist_mutation",
    "swap_table",
    "cache_stats",
    "clear_cache",
]

#: ``(static configuration) -> (graph closure, executable)``.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the mutation layer (graph memo + shared compile cache)."""
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop every memoized mutation graph."""
    _runtime.reset(_EXECUTABLE_MEMO)


def _check_kind(kind: str) -> str:
    if kind not in MUTATION_KINDS:
        raise ValueError(
            f"unknown mutation kind {kind!r}; the kinds are {MUTATION_KINDS}"
        )
    return kind


# --------------------------------------------------------------------------- #
# static tables
# --------------------------------------------------------------------------- #
def swap_table(tree_config: GPTreeConfig):
    """The same-arity function table of a config (numpy, ready for constants).

    Returns:
        ``(candidates, offsets, counts)`` -- ``candidates`` an ``int32`` vector
        of function ids concatenated arity by arity (1, then 2, then 3),
        ``offsets``/``counts`` ``int32[4]`` indexed by arity (arity 0 -- a leaf
        -- has no candidates).
    """
    grouped = {1: [], 2: [], 3: []}
    for name, _weight in tree_config.using_funcs:
        function_id = FUNC_IDS[name]
        grouped[FUNC_ARITY[function_id]].append(function_id)
    offsets = np.zeros(4, dtype=np.int32)
    counts = np.zeros(4, dtype=np.int32)
    flat = []
    for arity in (1, 2, 3):
        offsets[arity] = len(flat)
        counts[arity] = len(grouped[arity])
        flat.extend(grouped[arity])
    if not flat:  # pragma: no cover - a config always has one function
        raise ValueError("the config's function set is empty")
    return np.asarray(flat, dtype=np.int32), offsets, counts


# --------------------------------------------------------------------------- #
# trace-time operators
# --------------------------------------------------------------------------- #
def _point_traced(evoxc_module, tree_config, channels, key, gate):
    """One same-arity node edit per row (arity-preserving, ``size`` untouched)."""
    opcode, payload, size = channels
    pop_size, width = int(opcode.shape[0]), int(opcode.shape[1])
    candidates, offsets, counts = swap_table(tree_config)
    candidates_c = evoxc_module.constant(evoxc_module.from_numpy(candidates))
    offsets_c = evoxc_module.constant(evoxc_module.from_numpy(offsets))
    counts_c = evoxc_module.constant(evoxc_module.from_numpy(counts))
    arity_c = evoxc_module.constant(
        evoxc_module.from_numpy(np.asarray(FUNC_ARITY, dtype=np.int32))
    )
    constants = np.asarray(tree_config.const_samples, dtype=np.float32)
    constants_c = evoxc_module.constant(evoxc_module.from_numpy(constants))
    input_len = int(tree_config.input_len)

    k_pos, k_func, k_var, k_const = evoxc_module.random.split_n(key, 4)
    pos = evoxc_module.random.randint(
        k_pos, (pop_size,), 0, size[:, 0], dtype=evoxc_module.int32
    )
    node_opcode = _runtime.row_values(evoxc_module, opcode, pos)  # [pop, 1]
    node_size = _runtime.row_values(evoxc_module, size, pos)
    is_leaf = evoxc_module.equal(node_size, _runtime.i32(evoxc_module, 1))
    arity = evoxc_module.select(
        is_leaf,
        evoxc_module.broadcast(_runtime.i32(evoxc_module, 0), (pop_size, 1)),
        evoxc_module.gather(arity_c, node_opcode, axis=0),
    )
    count = evoxc_module.gather(counts_c, arity, axis=0)
    swap = evoxc_module.logical_and(
        evoxc_module.logical_not(is_leaf),
        evoxc_module.greater(count, _runtime.i32(evoxc_module, 0)),
    )

    # function nodes: a uniform draw inside the node's arity class
    pick = evoxc_module.random.randint(
        k_func, (pop_size, 1), 0,
        evoxc_module.maximum(count, _runtime.i32(evoxc_module, 1)),
        dtype=evoxc_module.int32,
    )
    replacement = evoxc_module.gather(
        candidates_c,
        evoxc_module.add(evoxc_module.gather(offsets_c, arity, axis=0), pick),
        axis=0,
    )
    # leaf nodes: a fresh payload of the leaf's own kind (VAR index / CONST value)
    new_var = evoxc_module.cast(
        evoxc_module.random.randint(
            k_var, (pop_size, 1), 0, input_len, dtype=evoxc_module.int32
        ),
        evoxc_module.float32,
    )
    new_const = evoxc_module.gather(
        constants_c,
        evoxc_module.random.randint(
            k_const, (pop_size, 1), 0, int(constants.size),
            dtype=evoxc_module.int32,
        ),
        axis=0,
    )
    is_const = evoxc_module.equal(
        node_opcode, _runtime.i32(evoxc_module, NType.CONST)
    )
    leaf_payload = evoxc_module.select(is_const, new_const, new_var)
    chosen_opcode = evoxc_module.select(swap, replacement, node_opcode)
    chosen_payload = evoxc_module.select(
        is_leaf, leaf_payload, _runtime.f32(evoxc_module, 0.0)
    )

    # the row is changed iff it is gated AND the drawn node has an edit available
    changed_row = evoxc_module.logical_and(
        evoxc_module.reshape(gate, (pop_size,)),
        evoxc_module.reshape(
            evoxc_module.logical_or(is_leaf, swap), (pop_size,)
        ),
    )
    edit = evoxc_module.logical_and(
        evoxc_module.equal(_runtime.columns(evoxc_module, width),
                           evoxc_module.reshape(pos, (pop_size, 1))),
        evoxc_module.reshape(changed_row, (pop_size, 1)),
    )
    new_opcode = evoxc_module.select(
        edit,
        evoxc_module.broadcast(chosen_opcode, (pop_size, width)),
        opcode,
    )
    new_payload = evoxc_module.select(
        edit,
        evoxc_module.broadcast(chosen_payload, (pop_size, width)),
        payload,
    )
    return (
        evoxc_module.cast(new_opcode, evoxc_module.int32),
        evoxc_module.cast(new_payload, evoxc_module.float32),
        evoxc_module.cast(size, evoxc_module.int32),
    )


def _subtree_traced(evoxc_module, tree_config, channels, key, gate):
    """Replace one subtree per row by a fresh random tree (the generation graph)."""
    opcode, size = channels[0], channels[2]
    pop_size = int(opcode.shape[0])
    k_pos, k_gen = evoxc_module.random.split_n(key, 2)
    pos = evoxc_module.random.randint(
        k_pos, (pop_size,), 0, size[:, 0], dtype=evoxc_module.int32
    )
    generated = encoding._generator_fn(tree_config, pop_size)
    donor_opcode, donor_payload, donor_size, _next = generated(k_gen)
    donor = (donor_opcode, donor_payload, donor_size)
    donor_pos = evoxc_module.broadcast(_runtime.i32(evoxc_module, 0), (pop_size,))
    return splice(evoxc_module, channels, donor, pos, donor_pos, gate)


def _hoist_traced(evoxc_module, channels, key, gate):
    """Shrink every row to the subtree at one drawn node (never overflows)."""
    opcode, payload, size = channels
    pop_size, width = int(opcode.shape[0]), int(opcode.shape[1])
    (k_pos,) = evoxc_module.random.split_n(key, 1)
    pos = evoxc_module.random.randint(
        k_pos, (pop_size,), 0, size[:, 0], dtype=evoxc_module.int32
    )
    subtree = _runtime.row_values(evoxc_module, size, pos)  # [pop, 1]
    columns = _runtime.columns(evoxc_module, width)
    source = evoxc_module.clamp(
        evoxc_module.add(columns, evoxc_module.reshape(pos, (pop_size, 1))),
        0,
        width - 1,
    )
    keep = evoxc_module.logical_and(
        evoxc_module.less(columns, subtree),
        evoxc_module.reshape(gate, (pop_size, 1)),
    )
    return (
        evoxc_module.cast(
            evoxc_module.select(
                keep, _runtime.take_per_row(evoxc_module, opcode, source), opcode
            ),
            evoxc_module.int32,
        ),
        evoxc_module.cast(
            evoxc_module.select(
                keep, _runtime.take_per_row(evoxc_module, payload, source), payload
            ),
            evoxc_module.float32,
        ),
        evoxc_module.cast(
            evoxc_module.select(
                keep, _runtime.take_per_row(evoxc_module, size, source), size
            ),
            evoxc_module.int32,
        ),
    )


def apply_mutation(evoxc_module, tree_config, channels, key, probability=None,
                   kind: str = "subtree"):
    """Apply ``kind`` to every row of ``channels`` (trace-time entry point).

    Args:
        evoxc_module: the imported ``evoxc`` module.
        tree_config: the :class:`~evogp.gp.config.GPTreeConfig` (function set,
            constant table, ``input_len``).
        channels: ``(opcode, payload, size)`` symbolic channels.
        key: a symbolic RNG key (split here into the operator's draws and the
            per-row gate).
        probability: per-row application probability; ``None`` applies to every
            row.
        kind: one of :data:`MUTATION_KINDS`.

    Returns:
        The mutated ``(opcode, payload, size)`` triple, channel dtypes cast
        back, structurally well-formed by construction.
    """
    _check_kind(kind)
    k_operator, k_gate = evoxc_module.random.split_n(key, 2)
    pop_size = int(channels[0].shape[0])
    gate = _runtime.gate_tensor(evoxc_module, k_gate, probability, pop_size)
    if kind == "point":
        return _point_traced(evoxc_module, tree_config, channels, k_operator, gate)
    if kind == "subtree":
        return _subtree_traced(evoxc_module, tree_config, channels, k_operator, gate)
    return _hoist_traced(evoxc_module, channels, k_operator, gate)


# --------------------------------------------------------------------------- #
# host-facing entry points
# --------------------------------------------------------------------------- #
def _mutation_fn(tree_config: GPTreeConfig, kind: str, probability):
    """The traced ``(opcode, payload, size, key) -> channels + key`` graph."""

    def operator(opcode, payload, size, key):
        evoxc_module = _runtime.evoxc()
        _draw, k_next = evoxc_module.random.split_n(key, 2)
        mutated = apply_mutation(
            evoxc_module, tree_config, (opcode, payload, size), _draw,
            probability, kind,
        )
        return mutated[0], mutated[1], mutated[2], k_next

    return operator


def _run_operator(
    config,
    state,
    key,
    kind: Optional[str],
    probability: Optional[float],
    gp_config,
    backend: str,
    device: str,
) -> GPState:
    """Shared implementation of the three host-facing mutation entry points."""
    tree_config, resolved_gp, pop_size = _runtime.resolve_configs(
        config, gp_config
    )
    if tree_config is None:
        raise ValueError(
            "mutation needs a GPTreeConfig (the function set, the constant "
            "table and input_len are what the operator draws from)"
        )
    opcode_array, payload_array, size_array = _runtime.channels_numpy(state)
    if size_array.ndim != 2:
        raise ValueError(
            f"channels must be 2-D [pop_size, max_nodes], got {size_array.shape}"
        )
    actual_pop, width = (int(dim) for dim in size_array.shape)
    if resolved_gp is not None and resolved_gp.pop_size != actual_pop:
        raise ValueError(
            f"the population has {actual_pop} rows but the config's pop_size is "
            f"{resolved_gp.pop_size}"
        )
    resolved_kind = kind
    if resolved_kind is None:
        resolved_kind = getattr(resolved_gp, "mutation_kind", "subtree")
    _check_kind(resolved_kind)
    if probability is not None and not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"probability must be in [0.0, 1.0], got {probability}")

    evoxc_module = _runtime.evoxc()
    key_tensor = _runtime.resolve_key(key)
    memo_key = (
        tree_config,
        resolved_kind,
        None if probability is None else float(probability),
        actual_pop,
        width,
        backend,
        device,
        tuple(int(size_) for size_ in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _mutation_fn(tree_config, resolved_kind, probability),
        (
            evoxc_module.TensorSpec((actual_pop, width), evoxc_module.int32),
            evoxc_module.TensorSpec((actual_pop, width), evoxc_module.float32),
            evoxc_module.TensorSpec((actual_pop, width), evoxc_module.int32),
            _runtime.key_spec(evoxc_module, key_tensor),
        ),
        backend,
        device,
    )
    opcode, payload, size, next_key = evoxc_module.run(
        built,
        opcode_array.astype(np.int32),
        payload_array.astype(np.float32),
        size_array.astype(np.int32),
        key_tensor,
    )
    return GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=_runtime.unevaluated(actual_pop),
        key=next_key,
    )


def mutate(
    config,
    state,
    key=0,
    *,
    kind: Optional[str] = None,
    probability: Optional[float] = None,
    gp_config=None,
    backend: str = "numpy",
    device: str = "cpu",
) -> GPState:
    """Mutate a population, returning OFFSPRING ONLY (elitism is the driver's).

    Args:
        config: a :class:`~evogp.gp.config.GPTreeConfig` or a
            ``(GPTreeConfig, GPConfig)`` pair.
        state: the population (a :class:`~evogp.gp.state.GPState` or an
            ``(opcode, payload, size)`` triple).
        key: int seed or a concrete key tensor.
        kind: ``"point"``, ``"subtree"`` or ``"hoist"`` (``None`` takes
            ``GPConfig.mutation_kind`` when a config pair is given).
        probability: per-row application probability in ``[0, 1]``; ``None``
            (default) applies to every row.  :func:`evogp.gp.variation` passes
            ``GPConfig.mutation_prob`` here.
        gp_config: the operator configuration when ``config`` is a bare tree
            config (only used for ``GPConfig.mutation_kind`` and the
            ``pop_size`` cross-check).
        backend: evoxc backend name.
        device: evoxc device string.

    Returns:
        A :class:`~evogp.gp.state.GPState` of the same shape with an
        all-``-inf`` fitness vector and the next RNG key.

    Raises:
        ValueError: on an unknown kind, a probability outside ``[0, 1]``, a
            missing tree config, or a population/config size mismatch.
    """
    return _run_operator(
        config, state, key, kind, probability, gp_config, backend, device
    )


def point_mutation(config, state, key=0, *, probability=None, gp_config=None,
                   backend: str = "numpy", device: str = "cpu") -> GPState:
    """Point mutation only (one same-arity node edit per selected row)."""
    return _run_operator(
        config, state, key, "point", probability, gp_config, backend, device
    )


def subtree_mutation(config, state, key=0, *, probability=None, gp_config=None,
                     backend: str = "numpy", device: str = "cpu") -> GPState:
    """Subtree-replacement mutation only (a fresh random subtree per row)."""
    return _run_operator(
        config, state, key, "subtree", probability, gp_config, backend, device
    )


def hoist_mutation(config, state, key=0, *, probability=None, gp_config=None,
                   backend: str = "numpy", device: str = "cpu") -> GPState:
    """Hoist mutation only (shrink to one drawn subtree)."""
    return _run_operator(
        config, state, key, "hoist", probability, gp_config, backend, device
    )
