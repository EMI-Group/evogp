"""SPIKE PROTOTYPE: one-graph "dense" population evaluator for the ETL engine.

This module is **additive and experimental** (spike round).  It is NOT wired
into :mod:`evogp.tree.forest` / :mod:`evogp.tree.tree` /
:mod:`evogp.tree.etl_engine` and nothing imports it by default; callers opt in
explicitly (or, later, through an env switch -- see "Integration proposal").
It exists to answer one question with numbers: **can the whole population be
evaluated by ONE compiled ETL executable fed with the population's node tensors
as RUNTIME DATA, instead of one compiled graph per tree structure?**

The specialized engine (:mod:`evogp.tree.etl_engine`) compiles one ETL graph per
*tree structure* and calls it once per tree: a ``pop_size`` fitness pass is
``pop_size`` ``build_cached`` lookups + ``pop_size`` ``etl.run`` calls +
``pop_size`` device readbacks (IREE), and a new structure costs a 0.4-2.6 s
compile there.  The dense prototype compiles ONE graph per *shape signature*
``(L, P, N, input_len, per_row, update_mode, backend, device, options)`` and
feeds the decoded population through ordinary tensor inputs, so the compile is
reused across calls, generations and populations of the same shape.

Scheme (see :func:`_dense_body` for the trace-time body)
-------------------------------------------------------
1. **Host preprocessing** (:func:`prepare_dense`) -- bulk-vectorized numpy over
   the ``(pop_size, max_tree_len)`` population tensors, no per-tree Python loop.
   It produces ``(L, P)``-shaped runtime *input* tensors: ``opcode`` (0 VAR,
   1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC/unknown-base, 5 inactive/padding),
   ``func_id`` (truncated toward zero, clamped to ``[0, 29]``), ``const_value``
   (float32), ``var_index`` (truncated, validated, clamped) and the three prefix
   child positions ``child1/2/3`` (clamped to ``[0, L)``).
   ``L = min(cap, max_tree_len)`` where ``cap`` is :data:`MAX_TREE_LEN_ENV`
   (default 32); a population whose longest tree exceeds ``L`` -- or that fails
   *any* structural check -- raises :class:`DenseUnsupported` so the caller falls
   back to the specialized path (which owns the error contract; this module
   deliberately does NOT reproduce
   :func:`etl_engine.decode_population`'s error messages).
2. **One graph** (module-level ``@etl.defn _dense_step``) with a Python
   trace-time ``for i in range(L-1, -1, -1)`` unroll over node positions
   (reverse prefix order: children always have a larger index than their
   parent).  Per position it gathers the three child values (single-axis
   ``gather`` from a flat view of the value buffer -- the documented flatten
   trick), computes the whole function-candidate chain with the *same* private
   emitters the specialized engine uses (:func:`etl_engine._emit_unary` /
   :func:`_emit_binary` / :func:`_emit_ternary`, parity by construction) and
   selects the leaf / function value with broadcast ``select`` preds.
3. **No per-tree host calls, no per-tree readbacks**: the nodes are inputs, the
   graph keeps one row of the value buffer per position, and the caller reads
   back the assembled result ONCE per chunk (numpy) or once per chunk from the
   device (IREE).

Value buffer & write modes (:data:`UPDATE_MODE_ENV`)
---------------------------------------------------
The trace keeps already-computed values as a buffer of shape ``(L, P, N)``
(``N == 1`` for ``per_row=True``).  ETL has no in-place mutation primitive that
is safe on every backend, so the row write is emulated:

``grow`` (default)
    The buffer starts as the row computed at ``i = L-1`` and is *prepended* to
    with ``concatenate([row, buf], axis=0)``; reads use the offset
    ``child - (i+1)``, clamped.  Row ``L-1`` is provably leaf-only for every
    validated tree (a function node there would need a child outside the row),
    so its three gathers are replaced by a constant.  Costs
    ``L*(L+1)/2 * P*N`` copied elements instead of ``L*L*P*N``, needs no initial
    buffer constant and uses only backend-agnostic ops.
``concat``
    A full ``(L, P, N)`` zero buffer (an embedded graph constant) written with
    ``concatenate([head, row, tail], axis=0)`` over static ``slice`` pieces --
    the iree-safe "select-mask" update without a boolean mask tensor.  Reads are
    plain ``child`` indices.
``scatter``
    A full zero buffer written with ``scatter(buf, i, row, axis=0)``.
    **numpy backend only by default**: on iree-cuda ETL's ``scatter`` mutates its
    input tensor in place (STOP-SHIP finding of the ETL investigation -- a
    retained input is modified by later invokes), so iree refuses unless
    :data:`ALLOW_UNSAFE_SCATTER_ENV` is set (the spike script uses it for the
    optional "is an in-graph-constructed buffer safe?" experiment).

Chunking / peak memory
----------------------
The graph shaves ``P`` and ``N`` into fixed chunk sizes so one compile serves
the whole call (the last chunk is **padded** to the chunk size -- one shape per
compile).  The documented model (:func:`dense_peak_bytes`) is::

    peak_bytes(L, P, N, mode) = 4 * (buf_factor*L*P*N + (LIVE + 8)*P*N + 4*L*P)

with ``buf_factor = 2`` for ``grow`` and ``3`` otherwise, and ``LIVE = 40`` live
``(P, N)`` SSA candidates (the 28 function candidates + 3 gathered children +
leaf/misc).  ``N`` is shaved first, then ``P``, so that ``peak_bytes <=
budget`` (:data:`BUDGET_MB_ENV`, default 2048 MB).  This is a *design* target for
the peak, not a measured RSS.

Environment knobs declared by this round
---------------------------------------
=============================== ======= ======================================
``EVOGP_DENSE_MAX_TREE_LEN``    32      max trace length ``L``; longer trees ->
                                        :class:`DenseUnsupported` (fallback)
``EVOGP_DENSE_BUDGET_MB``       2048    chunking memory budget (MiB)
``EVOGP_DENSE_UPDATE_MODE``     grow    ``grow`` / ``concat`` / ``scatter``
``EVOGP_DENSE_ALLOW_UNSAFE_SCATTER``    unset  permits ``scatter`` on iree
=============================== ======= ======================================
Backend/device/opt-level knobs, the shared compile cache
(:data:`DENSE_CACHE` is the very same :data:`etl_engine.CACHE`) and the
``ETL_DISABLE_LOCATIONS=1`` tracing speedup are inherited unchanged from
:mod:`evogp.tree.etl_engine`.

Parity notes / known numeric deviations (spike findings)
--------------------------------------------------------
* Function and leaf values go through the identical emission path, so a dense
  value is bit-exact unless the *specialized* engine folded a constant subtree at
  decode time (``etl_engine._decode_row`` -> ``_fold_*``).  A folded chain
  rounds to float32 after every op, exactly like the emitted graph, so the only
  way to diverge is a host ``numpy`` float64 primitive (``pow``/``exp``/``sin``/
  ... in the fold path) disagreeing by one ulp with the backend's float64
  primitive before the float32 rounding -- ``dense_spike_check.py --full``
  searches for such cases and reports every non-bit-exact finite case.
* ``output_len > 1`` (multi-output trees, OUT_NODE tags, the out-node
  accumulation protocol) is NOT supported this round: :class:`DenseUnsupported`.
* ``per_row=True`` -> ``(pop_size, output_len)`` and ``per_row=False`` ->
  ``(pop_size, N, output_len)`` follow :func:`etl_engine.evaluate`'s contract
  (fresh, contiguous, ``requires_grad=False``, float32, on ``x``'s device unless
  ``out_device``); :func:`sr_fitness_dense` mirrors
  :func:`etl_engine.sr_fitness` (positive loss ``sum_N sum_out err / N``, MSE or
  MAE, NaN/inf propagate) with the very same torch reduction ops.

Integration proposal (follow-up round; nothing here is wired yet)
----------------------------------------------------------------
* Wiring points: ``Forest.batch_forward`` / ``Forest.forward`` (``forest.py``)
  and ``Tree.forward`` (``tree.py``) would try :func:`evaluate_dense` first and
  fall back to ``etl_engine.evaluate(decode_population(...))`` on
  :class:`DenseUnsupported`; ``Forest.SR_fitness`` / ``Tree.SR_fitness``
  likewise around ``etl_engine.sr_fitness``.  The raw population tensors are
  already on hand there, so the dense path needs no decode.
* Env switch semantics: one master switch (``EVOGP_DENSE_ENGINE=off|auto|on``:
  ``off`` = specialized only, ``auto`` = dense when supported *and* ``L`` fits
  the cap *and* ``pop_size`` exceeds a measured threshold, ``on`` = dense for
  everything it supports) plus the knobs above.  ``auto`` needs the measured
  dense-vs-specialized crossover (``dense_spike_check.py``).
* The dense graphs live in the same ``etl_engine.CACHE``: one entry per shape
  signature, so ``etl_engine.cache_stats()`` already reports them and the
  4096-entry LRU stays cold.
* Follow-ups only if the measurements justify them: cache device-resident prep
  tensors and ``x`` across calls (a generation re-evaluates a NEW population on
  the SAME ``x``); gate the per-position candidate chain by the position's
  possible opcodes; blocked writes (several rows per buffer update) to cut the
  ``L^2`` buffer traffic; an in-graph fitness reduction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import etl

from .utils import NType
from . import etl_engine as _engine
from .etl_engine import (  # private but deliberately reused: parity by construction
    CACHE,
    _CPU,
    _as_numpy,
    _emit_binary,
    _emit_ternary,
    _emit_unary,
    _env_int,
    _operand,
    _quiet,
    _resolve_backend,
    _trace_const,
)

__all__ = [
    "ALLOW_UNSAFE_SCATTER_ENV",
    "BUDGET_MB_ENV",
    "DENSE_CACHE",
    "DEFAULT_BUDGET_MB",
    "DEFAULT_MAX_TREE_LEN",
    "DensePlan",
    "DensePrep",
    "DenseUnsupported",
    "MAX_TREE_LEN_ENV",
    "OP_BFUNC",
    "OP_CONST",
    "OP_INACTIVE",
    "OP_TFUNC",
    "OP_UFUNC",
    "OP_VAR",
    "UPDATE_MODES",
    "UPDATE_MODE_ENV",
    "build_dense",
    "dense_peak_bytes",
    "describe_plan",
    "evaluate_dense",
    "plan_dense",
    "prepare_dense",
    "sr_fitness_dense",
]

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: Max trace length ``L`` (``EVOGP_DENSE_MAX_TREE_LEN``): populations whose
#: longest tree exceeds it fall back to the specialized engine.  This is the
#: knob that bounds trace time and graph size (the candidate chain is emitted
#: ``L`` times) -- see the spike report's ``L`` scaling curve.
MAX_TREE_LEN_ENV = "EVOGP_DENSE_MAX_TREE_LEN"
#: Chunking memory budget in MiB (``EVOGP_DENSE_BUDGET_MB``).
BUDGET_MB_ENV = "EVOGP_DENSE_BUDGET_MB"
#: Buffer write strategy (``EVOGP_DENSE_UPDATE_MODE``): see the module docstring.
UPDATE_MODE_ENV = "EVOGP_DENSE_UPDATE_MODE"
#: Explicit opt-in for ``scatter`` on IREE, where ETL's scatter mutates its input.
ALLOW_UNSAFE_SCATTER_ENV = "EVOGP_DENSE_ALLOW_UNSAFE_SCATTER"

DEFAULT_MAX_TREE_LEN = 32
DEFAULT_BUDGET_MB = 2048
UPDATE_MODES = ("grow", "concat", "scatter")
DEFAULT_UPDATE_MODE = "grow"

#: The dense graphs live in the *same* compile cache as the specialized ones.
DENSE_CACHE = CACHE

#: ``opcode`` values (mirroring ``NType`` base values, plus 5 for padding).
OP_VAR = int(NType.VAR)  # 0
OP_CONST = int(NType.CONST)  # 1
OP_UFUNC = int(NType.UFUNC)  # 2 (arity 1)
OP_BFUNC = int(NType.BFUNC)  # 3 (arity 2)
OP_TFUNC = int(NType.TFUNC)  # 4 (arity 3 -- also every unknown base type)
OP_INACTIVE = 5  # padding / tree-length tail (all masks false -> value 0.0)

#: Highest function id the candidate chain dispatches on (``Func.LOOSE_SQRT``).
_MAX_FUNC_ID = 28
#: Highest binary (arity-2) function id (``Func.GE``); ids above it are unary.
_BFUNC_MAX = 13
#: Clamp for the truncated function id: ids ``>= 29`` are unmatched everywhere in
#: the engine and therefore yield 0.0; clamping keeps the int32 chain in range.
_FUNC_CLAMP = 29
#: Live ``(P, N)`` SSA candidates counted by :func:`dense_peak_bytes`.
_LIVE_CANDIDATES = 40


class DenseUnsupported(RuntimeError):
    """The dense path cannot (safely) evaluate this population.

    Callers must fall back to the specialized engine
    (:func:`etl_engine.decode_population` + :func:`etl_engine.evaluate`), which
    owns the error contract for malformed trees.  ``reason`` carries the
    human-readable cause; this module deliberately does not reproduce the
    specialized decoder's ``ValueError`` messages.
    """

    def __init__(self, reason: str):
        super().__init__(f"dense evaluation unsupported: {reason}")
        self.reason = reason


def _resolve_max_tree_len(value: Optional[int]) -> int:
    cap = (
        _env_int(MAX_TREE_LEN_ENV, DEFAULT_MAX_TREE_LEN)
        if value is None
        else int(value)
    )
    if cap <= 0:
        raise ValueError(f"max_tree_len={cap} must be positive")
    return cap


def _resolve_budget_mb(value: Optional[float]) -> float:
    if value is None:
        raw = os.environ.get(BUDGET_MB_ENV, "").strip()
        budget = float(raw) if raw else float(DEFAULT_BUDGET_MB)
    else:
        budget = float(value)
    if budget <= 0:
        raise ValueError(f"budget_mb={budget} must be positive")
    return budget


def resolve_update_mode(value: Optional[str], backend_name: Optional[str]) -> str:
    """Resolve ``EVOGP_DENSE_UPDATE_MODE`` (explicit argument wins)."""
    mode = value if value else os.environ.get(UPDATE_MODE_ENV, "").strip()
    if not mode:
        mode = DEFAULT_UPDATE_MODE
    mode = mode.strip().lower()
    if mode not in UPDATE_MODES:
        raise ValueError(
            f"unsupported update_mode {mode!r}: evogp.tree.dense_engine supports "
            f"{list(UPDATE_MODES)} (set {UPDATE_MODE_ENV!r} or pass update_mode=)"
        )
    if mode == "scatter" and backend_name == "iree":
        allowed = os.environ.get(ALLOW_UNSAFE_SCATTER_ENV, "").strip() not in ("", "0")
        if not allowed:
            # ETL's iree scatter mutates its input tensor in place and the value
            # buffer is retained across the trace-time loop, so a later invoke
            # would re-apply the row writes.
            raise DenseUnsupported(
                "update_mode='scatter' is unsafe on the iree backend (ETL's "
                "scatter mutates its input tensor in place); use "
                f"{UPDATE_MODE_ENV}=grow|concat or set "
                f"{ALLOW_UNSAFE_SCATTER_ENV}=1 to opt in for experiments"
            )
    return mode


# ---------------------------------------------------------------------------
# host preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensePrep:
    """Host preprocessing of one population, ready to be fed as graph inputs.

    Every array is ``(trace_len, pop_size)`` in prefix (pre-order) order; only
    positions ``i < tree_len[p]`` are meaningful and are marked by ``opcode !=
    OP_INACTIVE``.  The tail positions carry ``opcode == OP_INACTIVE`` and can
    never influence a real value: validation guarantees every child of a real
    node lies inside its own subtree, and padding rows never point outside
    ``[0, L)`` (clamped), so they only ever read other padding rows.
    """

    input_len: int
    output_len: int
    pop_size: int
    #: length of the population tensors' rows (``max_tree_len``)
    row_len: int
    #: ``L``: positions the graph unrolls (``min(cap, row_len)``)
    trace_len: int
    #: ``(pop_size,)`` declared tree lengths (``batch_subtree_size[:, 0]``)
    tree_len: np.ndarray
    #: ``(L, P)`` bool: position ``i`` belongs to tree ``p``
    real: np.ndarray
    opcode: np.ndarray
    func_id: np.ndarray
    const_value: np.ndarray
    var_index: np.ndarray
    child1: np.ndarray
    child2: np.ndarray
    child3: np.ndarray

    @property
    def max_tree_length(self) -> int:
        """Longest declared tree in this population."""
        return int(self.tree_len.max(initial=0))


def prepare_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    *,
    max_tree_len: Optional[int] = None,
) -> DensePrep:
    """Decode/validate a population for the dense graph (host-side, vectorized).

    Validation is a **superset** of :func:`etl_engine.decode_population`'s
    structural checks, applied only to positions that belong to a tree: a
    population the dense path accepts is one the specialized decoder accepts
    too, and anything doubtful raises :class:`DenseUnsupported`.

    Raises:
        DenseUnsupported: unsupported ``output_len``, an empty population, a tree
            longer than ``L = min(max_tree_len, row_len)``, an OUT_NODE tag, or
            any malformed structure (subtree size / child position / variable
            index / function id).
        TypeError, ValueError: the same shape/type checks the specialized
            decoder performs on its inputs (2-D tensors, matching shapes,
            ``input_len > 0``).
    """
    input_len = int(input_len)
    output_len = int(output_len)
    if input_len <= 0:
        raise ValueError(f"input_len={input_len} must be positive")
    if output_len <= 0:
        raise ValueError(f"output_len={output_len} must be positive")
    if output_len != 1:
        raise DenseUnsupported(
            f"output_len={output_len}: the dense spike implements the "
            "single-output protocol only"
        )
    values = _as_numpy(batch_node_value, np.float32, "batch_node_value")
    types = _as_numpy(batch_node_type, np.int16, "batch_node_type")
    sizes = _as_numpy(batch_subtree_size, np.int16, "batch_subtree_size")
    if values.ndim != 2:
        raise TypeError(
            "batch_node_value must be 2-D (pop_size, max_tree_len), got shape "
            f"{values.shape}"
        )
    if types.shape != values.shape or sizes.shape != values.shape:
        raise ValueError(
            "batch_node_value/batch_node_type/batch_subtree_size must have the "
            f"same shape, got {values.shape}, {types.shape}, {sizes.shape}"
        )
    pop_size, row_len = int(values.shape[0]), int(values.shape[1])
    if pop_size <= 0 or row_len <= 0:
        raise DenseUnsupported(f"empty population shape {values.shape}")

    cap = _resolve_max_tree_len(max_tree_len)
    L = min(cap, row_len)
    tree_len = sizes[:, 0].astype(np.int32)
    if (tree_len <= 0).any():
        raise DenseUnsupported("non-positive subtree_size[0] (tree length)")
    longest = int(tree_len.max())
    if longest > L:
        raise DenseUnsupported(
            f"longest tree is {longest} nodes but the dense trace cap is {L} "
            f"({MAX_TREE_LEN_ENV}={cap}, row length {row_len}); raise the cap or "
            "fall back to the specialized engine"
        )
    columns = _decode_columns(
        np.ascontiguousarray(values[:, :L].T),
        np.ascontiguousarray(types[:, :L].T),
        np.ascontiguousarray(sizes[:, :L].T).astype(np.int32),
        tree_len,
        input_len,
        L,
    )
    rows = np.arange(L, dtype=np.int32)[:, None]
    return DensePrep(
        input_len=input_len,
        output_len=output_len,
        pop_size=pop_size,
        row_len=row_len,
        trace_len=L,
        tree_len=tree_len,
        real=np.ascontiguousarray(rows < tree_len[None, :]),
        opcode=columns[0],
        func_id=columns[1],
        const_value=columns[2],
        var_index=columns[3],
        child1=columns[4],
        child2=columns[5],
        child3=columns[6],
    )


def _decode_columns(
    val_t: np.ndarray,
    type_t: np.ndarray,
    size_t: np.ndarray,
    tree_len: np.ndarray,
    input_len: int,
    L: int,
) -> Tuple[np.ndarray, ...]:
    """Vectorized node decode + structural validation of ``(L, P)`` node tables.

    The structural check is the vectorized form of the specialized decoder's
    reverse-prefix pass: for every node inside a tree, ``subtree_size`` must
    equal ``1 + sum(subtree_size[child])``, the first child of node ``i`` must
    be ``i+1``, the other children follow from the declared sizes
    (``c2 = i+1+s[c1]``, ``c3 = c2+s[c2]``), every child must stay inside the
    node's own subtree, and the subtree must stay inside the declared tree.  A
    valid prefix encoding is then exactly the stack order the specialized
    engine evaluates, so the child positions computed here are the real ones.
    """
    raw = type_t.astype(np.int32)
    base = raw & int(NType.TYPE_MASK)
    rows = np.arange(L, dtype=np.int32)[:, None]
    real = rows < tree_len[None, :]
    if ((raw & int(NType.OUT_NODE)) != 0)[real].any():
        raise DenseUnsupported(
            "an OUT_NODE-tagged node in a single-output tree (the specialized "
            "decoder raises ValueError here)"
        )
    # 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC/unknown base (arity 3)
    op = np.where(
        base <= 1, base, np.where(base == 2, 2, np.where(base == 3, 3, OP_TFUNC))
    ).astype(np.int32)
    leaf_real = real & (op <= 1)
    fn = real & (op >= 2)
    if (size_t[leaf_real] != 1).any():
        raise DenseUnsupported("a leaf node declares subtree_size != 1")

    # -- children (prefix encoding) ------------------------------------------
    c1_raw = rows + 1
    c1 = np.broadcast_to(np.minimum(c1_raw, L - 1), size_t.shape)
    s_c1 = np.take_along_axis(size_t, c1, axis=0)
    c2_raw = c1_raw + s_c1
    c2 = np.minimum(c2_raw, L - 1)
    s_c2 = np.take_along_axis(size_t, c2, axis=0)
    c3_raw = c2_raw + s_c2
    c3 = np.minimum(c3_raw, L - 1)
    s_c3 = np.take_along_axis(size_t, c3, axis=0)
    arity = np.where(op == OP_UFUNC, 1, np.where(op == OP_BFUNC, 2, 3)).astype(
        np.int32
    )
    need2 = fn & (arity >= 2)
    need3 = fn & (arity == 3)
    expected = 1 + s_c1 + np.where(need2, s_c2, 0) + np.where(need3, s_c3, 0)
    limit = rows + size_t - 1  # last position of this node's subtree
    bad = fn & (size_t != expected)
    bad |= fn & (c1_raw > limit)
    bad |= need2 & (c2_raw > limit)
    bad |= need3 & (c3_raw > limit)
    bad |= real & (rows + size_t > tree_len[None, :])
    if bad.any():
        raise DenseUnsupported(
            "malformed tree structure (subtree sizes / child positions are "
            "inconsistent)"
        )

    # -- leaves ---------------------------------------------------------------
    shape = size_t.shape
    opcode = np.where(real, op, OP_INACTIVE).astype(np.int32)
    func_id = np.zeros(shape, dtype=np.int32)
    const_value = np.zeros(shape, dtype=np.float32)
    var_index = np.zeros(shape, dtype=np.int32)
    var_real = real & (op == OP_VAR)
    if var_real.any():
        raw_var = val_t[var_real]
        if not np.isfinite(raw_var).all():
            raise DenseUnsupported("a non-finite variable index")
        indices = np.trunc(raw_var.astype(np.float64)).astype(np.int64)
        if (indices < 0).any() or (indices >= input_len).any():
            raise DenseUnsupported(f"a variable index outside [0, {input_len})")
        var_index[var_real] = indices.astype(np.int32)
    const_real = real & (op == OP_CONST)
    const_value[const_real] = val_t[const_real]

    # -- function ids (truncate toward zero, clamp into the dispatch range) ---
    if fn.any():
        raw_fn = val_t[fn].astype(np.float64)
        if not np.isfinite(raw_fn).all():
            raise DenseUnsupported("a non-finite function id")
        truncated = np.trunc(raw_fn)
        func_id[fn] = np.where(
            truncated < 0, 0, np.minimum(truncated, float(_FUNC_CLAMP))
        ).astype(np.int32)

    # -- child positions: in range everywhere (a padding row reads garbage, but
    #    it must never produce an out-of-range index -- iree poisons OOB) ------
    children = tuple(
        np.where(real, np.minimum(c, L - 1), 0).astype(np.int32) for c in (c1, c2, c3)
    )
    return (
        np.ascontiguousarray(opcode),
        np.ascontiguousarray(func_id),
        np.ascontiguousarray(const_value),
        np.ascontiguousarray(var_index),
        np.ascontiguousarray(children[0]),
        np.ascontiguousarray(children[1]),
        np.ascontiguousarray(children[2]),
    )


# ---------------------------------------------------------------------------
# the graph (module-level: build_cached keys on id(fn))
# ---------------------------------------------------------------------------


def _col(tensor: Any, i: int, P: int) -> Any:
    """Row ``i`` of an ``(L, P)`` runtime input as a ``(P, 1)`` symbolic tensor."""
    return etl.reshape(etl.slice(tensor, (i, 0), (1, P)), (P, 1))


def _read_buffer(
    buf: Any, rows: int, child: Any, offset_base: int, P: int, N: int, PN: int, pn_off: Any
) -> Any:
    """Gather row ``child[p]`` for every tree ``p`` -> ``(P, N)``.

    ``child`` is a clamped ``(P, 1)`` int32 tensor, ``rows`` the length of the
    buffer's leading axis and ``offset_base`` the shift that turns a node index
    into a buffer row (``i+1`` for the growing buffer, ``0`` for a full one).
    The resulting offset is clamped into ``[0, rows-1]`` so padding rows can
    never produce an out-of-range index.
    """
    off = etl.subtract(child, offset_base)
    off = etl.maximum(etl.minimum(off, rows - 1), 0)
    index = etl.reshape(etl.add(etl.multiply(off, PN), pn_off), (PN,))
    flat = etl.reshape(buf, (rows * PN,))
    return etl.reshape(etl.gather(flat, index, axis=0), (P, N))


def _dense_body(
    opcode: Any,
    func_id: Any,
    const_value: Any,
    var_index: Any,
    child1: Any,
    child2: Any,
    child3: Any,
    x: Any,
    cfg: Sequence[Any],
) -> Any:
    """Trace-time body of the dense evaluator (``cfg`` is a static tuple).

    ``cfg = (L, P, N, input_len, per_row, update_mode)``.  Returns the root
    values, ``(P, N)`` (``per_row=True`` uses ``N == 1``).
    """
    L = int(cfg[0])
    P = int(cfg[1])
    N = int(cfg[2])
    input_len = int(cfg[3])
    per_row = bool(cfg[4])
    mode = str(cfg[5])
    PN = P * N
    pn_off = etl.add(
        etl.reshape(etl.cast(etl.arange(P), etl.int32) * N, (P, 1)),
        etl.reshape(etl.cast(etl.arange(N), etl.int32), (1, N)),
    )
    zero = _trace_const(np.float32(0.0))
    if per_row:
        # x is (P, input_len): tree p reads its own row p.
        x_flat = etl.reshape(x, (P * input_len,))
        x_base = etl.reshape(etl.cast(etl.arange(P), etl.int32) * input_len, (P, 1))
    else:
        # x is (N, input_len): every tree reads all N rows.
        x_flat = etl.reshape(x, (N * input_len,))
        x_base = etl.reshape(etl.cast(etl.arange(N), etl.int32) * input_len, (1, N))

    if mode in ("concat", "scatter"):
        buf = etl.constant(etl.core.tensor(np.zeros((L, P, N), dtype=np.float32)))
    else:
        buf = None
    buf_rows = 0
    root = None

    for i in range(L - 1, -1, -1):
        op_i = _col(opcode, i, P)
        f_i = _col(func_id, i, P)
        cv_i = _col(const_value, i, P)
        vi_i = _col(var_index, i, P)
        c1_i = _col(child1, i, P)
        c2_i = _col(child2, i, P)
        c3_i = _col(child3, i, P)

        # -- child values ------------------------------------------------------
        if buf is None:
            # grow mode at i == L-1: a function node here would need a child
            # outside the row, which decode-time validation rejects, so the three
            # gathers are dead code and are replaced by a constant.
            v1 = v2 = v3 = _trace_const(np.zeros((P, 1), dtype=np.float32))
            has_children = False
        elif mode == "grow":
            v1 = _read_buffer(buf, buf_rows, c1_i, i + 1, P, N, PN, pn_off)
            v2 = _read_buffer(buf, buf_rows, c2_i, i + 1, P, N, PN, pn_off)
            v3 = _read_buffer(buf, buf_rows, c3_i, i + 1, P, N, PN, pn_off)
            has_children = True
        else:
            v1 = _read_buffer(buf, L, c1_i, 0, P, N, PN, pn_off)
            v2 = _read_buffer(buf, L, c2_i, 0, P, N, PN, pn_off)
            v3 = _read_buffer(buf, L, c3_i, 0, P, N, PN, pn_off)
            has_children = True

        # -- leaf value --------------------------------------------------------
        xidx = etl.reshape(etl.add(vi_i, x_base), (PN,))
        xv = etl.reshape(etl.gather(x_flat, xidx, axis=0), (P, N))
        leaf = etl.select(etl.equal(op_i, OP_VAR), xv, cv_i)

        if not has_children:
            val = leaf
        else:
            # -- function value: the whole candidate chain, engine emitters -----
            prod = zero
            for fid in range(_MAX_FUNC_ID, 0, -1):
                candidate = (
                    _emit_binary(fid, v1, v2)
                    if fid <= _BFUNC_MAX
                    else _emit_unary(fid, v1)
                )
                prod = etl.select(etl.equal(f_i, fid), _operand(candidate), prod)
            valid = etl.logical_or(
                etl.logical_and(
                    etl.equal(op_i, OP_BFUNC),
                    etl.logical_and(
                        etl.greater_equal(f_i, 1), etl.less_equal(f_i, _BFUNC_MAX)
                    ),
                ),
                etl.logical_and(
                    etl.equal(op_i, OP_UFUNC),
                    etl.greater_equal(f_i, _BFUNC_MAX + 1),
                ),
            )
            res = etl.select(valid, prod, zero)
            res = etl.select(etl.equal(op_i, OP_TFUNC), _emit_ternary(v1, v2, v3), res)
            val = etl.select(etl.less_equal(op_i, OP_CONST), leaf, res)

        if i == 0:
            root = val
            break

        # -- write row i -------------------------------------------------------
        row = etl.reshape(val, (1, P, N))
        if mode == "grow":
            buf = row if buf is None else etl.concatenate([row, buf], axis=0)
            buf_rows += 1
        elif mode == "concat":
            pieces = []
            if i > 0:
                pieces.append(etl.slice(buf, (0, 0, 0), (i, P, N)))
            pieces.append(row)
            if L - i - 1 > 0:
                pieces.append(etl.slice(buf, (i + 1, 0, 0), (L - i - 1, P, N)))
            buf = (
                pieces[0]
                if len(pieces) == 1
                else etl.concatenate(pieces, axis=0)
            )
        else:  # scatter -- numpy only by default (see resolve_update_mode)
            index = etl.constant(etl.core.tensor(np.asarray([i], dtype=np.int32)))
            buf = etl.scatter(buf, index, row, axis=0)
    return root


@etl.defn
def _dense_step(
    opcode, func_id, const_value, var_index, child1, child2, child3, x, cfg
):
    """One compiled dense population pass (module-level: ``id(fn)`` is the key).

    The trailing ``cfg`` is a static tuple (it specializes the graph and
    participates in the ``build_cached`` key); every other argument is a runtime
    tensor input whose *values* never trigger a recompile.
    """
    return _dense_body(
        opcode, func_id, const_value, var_index, child1, child2, child3, x, cfg
    )


# ---------------------------------------------------------------------------
# chunking & plan
# ---------------------------------------------------------------------------

#: live ``(P, N)`` intermediates added per slot by :func:`dense_peak_bytes`
_PN_EXTRA = _LIVE_CANDIDATES + 8


def dense_peak_bytes(
    L: int, P: int, N: int, update_mode: str = DEFAULT_UPDATE_MODE
) -> int:
    """Documented peak-memory model of one dense chunk (see the module docstring).

    ``4 * (buf_factor*L*P*N + (LIVE+8)*P*N + 4*L*P)`` float32 bytes with
    ``buf_factor = 2`` for ``grow`` (the buffer is rebuilt by concatenation, so
    one extra copy is live) and ``3`` otherwise (initial buffer + one full-size
    update temporary).
    """
    buf_factor = 2 if update_mode == "grow" else 3
    elements = buf_factor * L * P * N + _PN_EXTRA * P * N + 4 * L * P
    return int(4 * elements)


def _chunk_shape(
    L: int, P: int, N: int, update_mode: str, budget_bytes: float
) -> Tuple[int, int]:
    """Pick ``(P_chunk, N_chunk)`` so :func:`dense_peak_bytes` fits the budget."""
    if dense_peak_bytes(L, P, N, update_mode) <= budget_bytes:
        return P, N
    per_pn = 4 * ((2 if update_mode == "grow" else 3) * L + _PN_EXTRA)
    max_pn = max(1, int(budget_bytes // per_pn))
    n_chunk = max(1, min(N, max_pn // max(1, P)))
    p_chunk = P
    if P * n_chunk > max_pn:
        p_chunk = max(1, min(P, max_pn))
    return p_chunk, n_chunk


@dataclass(frozen=True)
class DensePlan:
    """Everything :func:`build_dense` / :func:`evaluate_dense` need.

    ``specs`` is the exact ``build_cached`` spec tuple (runtime
    :class:`etl.TensorSpec` leaves + the static ``cfg``); ``cfg`` alone is what
    :func:`etl.run` must receive as the trailing static argument.
    """

    prep: DensePrep
    per_row: bool
    #: the graph's ``N`` (``1`` for ``per_row=True``, else the data-point count)
    n_rows: int
    #: ``x``'s row count for the WHOLE call (``pop_size`` / data-point count)
    x_rows: int
    p_chunk: int
    n_chunk: int
    update_mode: str
    budget_mb: float
    input_len: int
    cfg: Tuple[Any, ...] = field(repr=False)
    specs: Tuple[Any, ...] = field(repr=False)

    @property
    def trace_len(self) -> int:
        return self.prep.trace_len

    @property
    def pop_size(self) -> int:
        return self.prep.pop_size

    @property
    def p_chunks(self) -> int:
        return -(-self.pop_size // self.p_chunk)

    @property
    def n_chunks(self) -> int:
        return -(-self.n_rows // self.n_chunk)

    @property
    def peak_bytes(self) -> int:
        return dense_peak_bytes(
            self.trace_len, self.p_chunk, self.n_chunk, self.update_mode
        )


def plan_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    n_rows: int,
    *,
    per_row: bool,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
    backend: Optional[str] = None,
) -> DensePlan:
    """Preprocess the population, pick the chunk shape and build the specs.

    ``n_rows`` is ``x``'s row count in the :func:`etl_engine.evaluate` sense
    (``pop_size`` when ``per_row=True``, the data-point count otherwise).  The
    graph's own ``N`` is ``1`` for ``per_row=True``; chunking then shaves ``P``.
    """
    prep = prepare_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        max_tree_len=max_tree_len,
    )
    backend_name = (
        backend if backend else os.environ.get(_engine.BACKEND_ENV, "").strip()
    )
    mode = resolve_update_mode(update_mode, backend_name or None)
    budget = _resolve_budget_mb(budget_mb)
    x_rows = int(n_rows) if per_row else max(1, int(n_rows))
    if per_row:
        x_rows = prep.pop_size
    graph_n = 1 if per_row else x_rows
    p_chunk, n_chunk = _chunk_shape(
        prep.trace_len,
        prep.pop_size,
        graph_n,
        mode,
        budget * 1024.0 * 1024.0,
    )
    shape = (prep.trace_len, p_chunk)
    x_shape = (
        (p_chunk, prep.input_len) if per_row else (n_chunk, prep.input_len)
    )
    cfg = (prep.trace_len, p_chunk, n_chunk, prep.input_len, bool(per_row), mode)
    specs: Tuple[Any, ...] = (
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(shape, etl.float32),
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(shape, etl.int32),
        etl.TensorSpec(x_shape, etl.float32),
        cfg,
    )
    return DensePlan(
        prep=prep,
        per_row=bool(per_row),
        n_rows=graph_n,
        x_rows=x_rows,
        p_chunk=p_chunk,
        n_chunk=n_chunk,
        update_mode=mode,
        budget_mb=budget,
        input_len=prep.input_len,
        cfg=cfg,
        specs=specs,
    )


def describe_plan(plan: DensePlan) -> Dict[str, Any]:
    """Flat, JSON-friendly summary of a plan (used by the spike script)."""
    return {
        "trace_len": plan.trace_len,
        "max_tree_length": plan.prep.max_tree_length,
        "row_len": plan.prep.row_len,
        "pop_size": plan.pop_size,
        "input_len": plan.input_len,
        "per_row": plan.per_row,
        "n_rows": plan.n_rows,
        "update_mode": plan.update_mode,
        "p_chunk": plan.p_chunk,
        "n_chunk": plan.n_chunk,
        "p_chunks": plan.p_chunks,
        "n_chunks": plan.n_chunks,
        "peak_bytes": plan.peak_bytes,
    }


def build_dense(
    plan: DensePlan,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> Any:
    """Build-or-hit the ONE dense executable of ``plan`` (shared compile cache).

    Mirrors :func:`etl_engine._executable`: the same in-repo ``build_cached``
    pattern, the same shared cache, and the same explicit backend/device/options
    resolution (the ambient ``ETL_*`` env cannot change it).  The runtime tensor
    inputs are keyed once on their ``TensorSpec``, so new populations of the same
    shape never recompile.
    """
    backend_name, device, options = _resolve_backend(backend, iree_device_index)
    if backend_name == "numpy":
        return etl.build_cached(
            _dense_step,
            *plan.specs,
            backend=backend_name,
            device=device,
            cache=DENSE_CACHE,
        )
    return etl.build_cached(
        _dense_step,
        *plan.specs,
        backend=backend_name,
        device=device,
        cache=DENSE_CACHE,
        **options,
    )


# ---------------------------------------------------------------------------
# runtime: chunked execution
# ---------------------------------------------------------------------------


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
    """Pad ``array`` (axis 0) up to ``rows`` rows by repeating its first row."""
    current = array.shape[0]
    if current >= rows:
        return array
    if current == 0:
        return np.zeros((rows,) + array.shape[1:], dtype=array.dtype)
    return np.concatenate(
        [array, np.repeat(array[:1], rows - current, axis=0)], axis=0
    )


def _prep_columns(prep: DensePrep, p0: int, p_chunk: int) -> Tuple[np.ndarray, ...]:
    """``(L, p_chunk)`` contiguous slices of every runtime input of one chunk."""
    columns: List[np.ndarray] = []
    for array in (
        prep.opcode,
        prep.func_id,
        prep.const_value,
        prep.var_index,
        prep.child1,
        prep.child2,
        prep.child3,
    ):
        chunk = array[:, p0 : p0 + p_chunk]
        columns.append(np.ascontiguousarray(_pad_rows(chunk.T, p_chunk).T))
    return tuple(columns)


def run_plan(
    plan: DensePlan,
    executable: Any,
    x: np.ndarray,
    *,
    backend_name: str,
    device: Any,
) -> np.ndarray:
    """Run every chunk of ``plan`` and assemble the ``(P, N)`` result (host).

    The result is the transposed / per-tree layout of
    :func:`etl_engine.evaluate`'s return value: ``(P, 1)`` rows for
    ``per_row=True``, ``(P, N)`` otherwise.
    """
    prep = plan.prep
    out = np.empty((prep.pop_size, plan.n_rows), dtype=np.float32)
    to_device = backend_name != "numpy"
    for p0 in range(0, prep.pop_size, plan.p_chunk):
        p_chunk = min(plan.p_chunk, prep.pop_size - p0)
        columns = _prep_columns(prep, p0, plan.p_chunk)
        n_chunks = 1 if plan.per_row else plan.n_chunks
        for chunk_index in range(n_chunks):
            n0 = 0 if plan.per_row else chunk_index * plan.n_chunk
            n_chunk = min(plan.n_chunk, plan.n_rows - n0) if not plan.per_row else 1
            if plan.per_row:
                xin = np.ascontiguousarray(x[p0 : p0 + p_chunk])
                xin = _pad_rows(xin, plan.p_chunk)
            else:
                xin = np.ascontiguousarray(x[n0 : n0 + n_chunk])
                xin = _pad_rows(xin, plan.n_chunk)
            if to_device:
                args = tuple(etl.from_numpy(a).to(device) for a in columns) + (
                    etl.from_numpy(xin).to(device),
                    plan.cfg,
                )
            else:
                args = columns + (xin, plan.cfg)
            with _quiet():
                result = etl.run(executable, *args)
            if to_device:
                host = np.asarray(result.to(_CPU).numpy(), dtype=np.float32)
            else:
                host = np.asarray(result.numpy(), dtype=np.float32)
            host = host.reshape(plan.p_chunk, plan.n_chunk)
            if plan.per_row:
                out[p0 : p0 + p_chunk, 0] = host[:p_chunk, 0]
            else:
                out[p0 : p0 + p_chunk, n0 : n0 + n_chunk] = host[
                    :p_chunk, :n_chunk
                ]
    return out


def _input_to_numpy(tensor: Any, input_len: int, name: str) -> np.ndarray:
    """``(N, input_len)`` float32 host copy of ``tensor`` (mirrors the engine)."""
    array = _as_numpy(tensor, np.float32, name)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (rows, {input_len}), got shape {array.shape}"
        )
    if array.shape[1] != input_len:
        raise ValueError(
            f"{name} has {array.shape[1]} columns but the population expects "
            f"input_len={input_len}"
        )
    return array


def evaluate_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
) -> torch.Tensor:
    """Evaluate a whole population with ONE dense graph (see module docstring).

    Contracts (mirroring :func:`etl_engine.evaluate` exactly):

    * ``per_row=True``: ``x`` is ``(pop_size, input_len)``, tree ``i`` sees row
      ``i``; the result is ``(pop_size, output_len)``.
    * ``per_row=False``: ``x`` is ``(N, input_len)``, every tree is evaluated on
      all ``N`` rows; the result is ``(pop_size, N, output_len)``.
    * The result is fresh, contiguous, ``requires_grad=False`` float32 on ``x``'s
      device (``out_device`` overrides).  ``x`` and the population tensors are
      never modified.

    Raises:
        DenseUnsupported: the population cannot be handled densely -- callers
            fall back to ``etl_engine.decode_population`` + ``evaluate``.
    """
    query = _as_numpy(x, np.float32, "x")
    if query.ndim != 2:
        raise ValueError(
            f"x must be 2-D (rows, {int(input_len)}), got shape {query.shape}"
        )
    plan = plan_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        int(query.shape[0]),
        per_row=per_row,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
    )
    if query.shape[1] != plan.input_len:
        raise ValueError(
            f"x has {query.shape[1]} columns but the population expects "
            f"input_len={plan.input_len}"
        )
    if per_row and query.shape[0] != plan.pop_size:
        raise ValueError(
            f"per_row=True needs x.shape[0] == {plan.pop_size} (pop_size), got "
            f"{query.shape[0]}"
        )
    executable = build_dense(
        plan, backend=backend, iree_device_index=iree_device_index
    )
    backend_name, device, _ = _resolve_backend(backend, iree_device_index)
    rows = run_plan(plan, executable, query, backend_name=backend_name, device=device)
    if per_row:
        rows = np.ascontiguousarray(rows[:, :1])
    else:
        rows = rows[:, :, None]
    tensor = torch.from_numpy(rows).contiguous().detach().requires_grad_(False)
    target = out_device if out_device is not None else getattr(x, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


def sr_fitness_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    inputs: Any,
    labels: Any,
    use_MSE: bool = True,
    *,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
) -> torch.Tensor:
    """Symbolic-regression fitness via the dense evaluator.

    Same contract as :func:`etl_engine.sr_fitness`: ``inputs`` is
    ``(N, input_len)``, ``labels`` is ``(N, output_len)`` and the result is a
    ``(pop_size,)`` float32 POSITIVE loss ``sum_N sum_out err / N`` (MSE or MAE)
    on ``inputs``' device unless ``out_device``; NaN/inf propagate.  The
    reduction uses the very same torch ops as the engine's version, so identical
    predictions give identical losses.
    """
    xt = _input_to_numpy(inputs, int(input_len), "inputs")
    if xt.shape[0] <= 0:
        raise ValueError("sr_fitness needs at least one data point, got 0 rows")
    label = _as_numpy(labels, np.float32, "labels")
    if label.ndim != 2 or label.shape != (xt.shape[0], int(output_len)):
        raise ValueError(
            f"labels must have shape ({xt.shape[0]}, {int(output_len)}), got "
            f"{label.shape}"
        )
    prediction = evaluate_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        int(input_len),
        int(output_len),
        inputs,
        per_row=False,
        out_device=out_device,
        backend=backend,
        iree_device_index=iree_device_index,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
    )
    target = torch.from_numpy(label).to(prediction.device)
    diff = target.unsqueeze(0) - prediction
    err = diff * diff if use_MSE else diff.abs()
    loss = err.sum(dim=(1, 2)) / float(xt.shape[0])
    return loss.contiguous().detach().requires_grad_(False)
