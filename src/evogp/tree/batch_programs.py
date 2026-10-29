"""Host-side program handling of the batch route: decode, dedup, trace, merge.

Everything between a population's three tensors and ONE merged
:class:`evoxc.Graph` lives here: the input / label conversions, the per-function
graph bodies, the structural dedup, the ``evoxc.trace`` / ``evoxc.batch_functions``
calls with their two structural exception types converted into
:class:`evogp.tree.batch_knobs.BatchUnsupported`, and :func:`_plan_host`, which
produces the :class:`BatchPlan` of one call (the frozen programs, the row mapping,
the traced graphs and the merged graph).

No compile, no front table and no torch tensor assembly happen here -- that is
:mod:`evogp.tree.batch_engine`, which imports this module and re-exports every
name in it (this module never imports ``batch_engine`` back at module level, so
``evogp.tree.batch_engine`` stays the public facade).

Two invariants of this module:

* a per-function graph is traced through a Python CLOSURE over the tree's static
  program, never with the program as a traced static argument: static values are
  part of a graph's identity, so a static program would make
  ``graph.static_values`` differ per tree and ``evoxc.batch_functions`` would raise
  ``TraceError`` on the merge;
* every "this route cannot (safely) handle it" answer (a shape doubt, an op of the
  merged graph without a batching rule, a ``TraceError`` / ``TransformError``) is a
  :class:`BatchUnsupported` decline, never a ``ValueError``: the specialized engine
  owns the malformed-tree error contract, so this route must not pre-empt it.

Decoding, validation, constant folding and op emission are the frozen engine's
(:func:`evogp.tree.evoxc_engine.decode_population` and
:func:`evogp.tree.evoxc_engine._walk_tokens`, which is the emitter *and* the
multi-output protocol) -- reuse, never re-implement.  Scheme and measured
economics: ``BATCH_EVAL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc
from evoxc.transforms import batching as _batching

from .batch_knobs import (
    LOSS_VARIANT,
    BatchUnsupported,
    _normalize_variant,
    resolve_alignment,
    resolve_band_count,
    resolve_switch_ops,
)
from .evoxc_engine import (
    _bits_to_float,
    _trace_const_col,
    _walk_tokens,
    decode_population,
)

# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
def _pop_size(tensor: Any) -> Optional[int]:
    """``(pop_size, max_tree_len)`` row count, or ``None`` when unreadable.

    Read from ``shape`` only (no host copy); ``None`` means "let
    ``decode_population`` raise its documented error", so an exotic or malformed
    population tensor is never turned into a silent decline.
    """
    shape = getattr(tensor, "shape", None)
    try:
        if shape is None or len(shape) != 2:
            return None
        return int(shape[0])
    except Exception:  # pragma: no cover - exotic tensor subclasses only
        return None


def _as_host(tensor: Any, name: str) -> np.ndarray:
    """Read any tensor (torch on any device / numpy) as a float32 host array."""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(tensor)
        except Exception as exc:  # pragma: no cover - exotic inputs only
            raise BatchUnsupported(
                f"batch_engine: {name} is not convertible to a host array "
                f"({type(tensor).__name__})"
            ) from exc
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _host_rows(tensor: Any, input_len: int, name: str, *, per_row: bool,
               pop_size: Optional[int]) -> np.ndarray:
    """Validate/convert a ``(rows, input_len)`` float32 input; never mutates it.

    Every shape doubt is a :class:`BatchUnsupported` decline (never a
    ``ValueError``): the fallback engine raises the documented errors, and a route
    that declares "this is not mine" must not pre-empt that contract.
    """
    array = _as_host(tensor, name)
    if array.ndim != 2:
        raise BatchUnsupported(
            f"batch_engine: {name} must be 2-D (rows, {int(input_len)}), got shape "
            f"{array.shape}"
        )
    if array.shape[1] != int(input_len):
        raise BatchUnsupported(
            f"batch_engine: {name} has {array.shape[1]} columns but the "
            f"population expects input_len={int(input_len)}"
        )
    if per_row and pop_size is not None and array.shape[0] != pop_size:
        raise BatchUnsupported(
            f"batch_engine: per_row=True needs {name}.shape[0] == {pop_size} "
            f"(pop_size), got {array.shape[0]}"
        )
    return array


def _host_labels(labels: Any, n_rows: int, output_len: int) -> np.ndarray:
    """Validate/convert the ``(n_rows, output_len)`` SR labels."""
    array = _as_host(labels, "labels")
    if array.shape != (int(n_rows), int(output_len)):
        raise BatchUnsupported(
            f"batch_engine: labels must have shape ({int(n_rows)}, "
            f"{int(output_len)}), got {array.shape}"
        )
    return array


# ---------------------------------------------------------------------------
# per-function graph bodies (the engine's own emitters -- parity by construction)
# ---------------------------------------------------------------------------
def _constant_output(const_out: Sequence[int], n_rows: int) -> Any:
    """The ``(n_rows, 1)`` constant column of a fully constant tree.

    A constant tree still has to RETURN a tensor of the same rank/shape as every
    other function's output (a bare 0-d ``evoxc.constant`` would be a rank
    mismatch and a static output leaf, which the merge rejects).  The column is
    built the way the specialized engine assembles a constant output
    (:func:`evogp.tree.evoxc_engine._trace_const_col`), i.e. as a real
    ``(n_rows, 1)`` payload; the merge stacks differing payloads into one stacked
    constant (``report["stacked_constants"]``), so it stays exact.
    """
    if len(const_out) == 1:
        return _trace_const_col(n_rows, _bits_to_float(const_out[0]))
    cols = [_trace_const_col(n_rows, _bits_to_float(bits)) for bits in const_out]
    return evoxc.concatenate(cols, axis=1)


def _predict(program: Any, const_out: Any, x: Any, n_rows: int) -> Any:
    """The ``(n_rows, out_len)`` symbolic output of one decoded tree."""
    if program is not None:
        return _walk_tokens(program[1], int(program[0]), x)
    return _constant_output(const_out, n_rows)


def _value_fn(program: Any, const_out: Any, n_rows: int):
    """One value function: ``x -> (n_rows, out_len)`` (a closure over the program).

    The program travels in the CLOSURE, never as a ``trace`` static argument:
    static values are part of the graph's identity, so a static program would give
    every function different ``graph.static_values`` and
    ``evoxc.batch_functions`` would raise ``TraceError`` on the merge.
    """

    def fn(x):
        return _predict(program, const_out, x, n_rows)

    return fn


def _loss_fn(program: Any, const_out: Any, n_rows: int, use_mse: bool):
    """One SR-loss function: ``(x, label) -> (1,)``-free RANK-0 scalar loss.

    The whole fitness reduction is IN-GRAPH, so the merged output is one scalar
    per function (``(n_functions,)``) and only that crosses back::

        err  = (pred - label) ** 2      if use_mse
        err  = |pred - label|           otherwise
        loss = sum_N sum_out err / N

    Outputs are summed and never divided by ``output_len`` (the SR contract), and
    the ``/ N`` divide is a static float32 constant inside the graph.  Rank-0
    per-function outputs turned out to work under ``out_axes=0`` (the merged result
    is ``(n_functions,)`` directly), so no host-side squeeze is needed; the
    reduction is done with :func:`evoxc.reduce_sum` over both axes, whose batching
    rule is registered.
    """

    def fn(x, label):
        pred = _predict(program, const_out, x, n_rows)
        diff = evoxc.subtract(pred, label)
        err = evoxc.multiply(diff, diff) if use_mse else evoxc.abs(diff)
        total = evoxc.reduce_sum(err, axes=(0, 1))
        return evoxc.divide(total, _scalar(n_rows))

    return fn


def _scalar(value: float) -> Any:
    """A rank-0 float32 symbolic constant (``evoxc`` rejects numpy scalars).

    Must be called inside a trace; a rank-0 payload is shared by every merged
    function (the divide's divisor is the same constant for all of them).
    """
    return evoxc.constant(
        evoxc.core.tensor(np.asarray(np.float32(value), dtype=np.float32))
    )


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchPlan:
    """The host-side plan of one merged batch graph (no executable, no cache).

    Attributes:
        input_len: problem input width.
        output_len: outputs per tree.
        pop_size: population size (the number of rows the plan answers for).
        variant: :data:`VALUE_VARIANT` or :data:`LOSS_VARIANT`.
        use_mse: the loss variant's error metric (ignored by the value variant).
        per_row: ``True`` when every population row has its own ``x`` row.
        n_rows: rows each merged function sees (``1`` for ``per_row``, else ``N``
            for the value variant and the data-point count for the loss variant).
        programs: per population row, the ``(program, const_out)`` identity of that
            row -- ``program`` is the frozen static program (``None`` for a fully
            constant tree, whose answer is then ``const_out``).
        functions: the DEDUPED ``(program, const_out)`` multiset actually merged, in
            first-appearance order: ``functions[row_index[row]] == programs[row]``.
        row_index: per population row, its index into :attr:`functions`.
        function_rows: per merged function, the population row it was first seen
            at (the ``x`` row a ``per_row`` plan feeds it).
        band_count: number of merge bands (``1`` = the flat merge).
        switch_ops: whether op-switching was requested (forces ``band_count == 1``).
        alignment: the ``switch_ops`` class-alignment mode.
        graphs: the traced per-function :class:`evoxc.Graph` objects.
        merged: the merged :class:`evoxc.Graph` (``merged.batch_report`` describes it).
        specs: the merged graph's tensor input specs (``evoxc.TensorSpec`` objects).
        report: ``merged.batch_report.as_dict()`` -- ``n_functions``,
            ``emitted_ops``, ``bands``, ``band_sizes``, ``band_strategy``, ...
        backend_name: the backend the plan was resolved for (the default of
            :func:`build_batch`).
        device: the resolved ``evoxc.core.Device``.
    """

    input_len: int
    output_len: int
    pop_size: int
    variant: str
    use_mse: bool
    per_row: bool
    n_rows: int
    programs: Tuple[Any, ...]
    functions: Tuple[Any, ...]
    row_index: Tuple[int, ...]
    function_rows: Tuple[int, ...]
    band_count: int
    switch_ops: bool
    alignment: str
    graphs: Tuple[Any, ...]
    merged: Any
    specs: Tuple[Any, ...]
    report: Dict[str, Any]
    backend_name: str
    device: Any


@dataclass(frozen=True)
class BatchExecutable:
    """A plan plus its compiled executable (the value of both front tables).

    Attributes:
        plan: the :class:`BatchPlan` this executable was built for.
        executable: the loaded ``evoxc.Executable`` of the merged graph.
        backend: the backend name it was compiled for.
        device: the ``evoxc.core.Device`` it was loaded on.
        pins: strong references to the population/input storages the plan memo entry
            is keyed on (empty for an unmemorized build); they only keep the
            ``data_ptr`` identity tokens ABA-proof, so they are never inspected.
    """

    plan: BatchPlan
    executable: Any
    backend: str
    device: Any
    pins: Tuple[Any, ...] = ()


def _dedup(pop: Any, x_host: np.ndarray, per_row: bool) -> Tuple[
    Tuple[Any, ...], Tuple[int, ...], Tuple[int, ...]
]:
    """Collapse identical rows: ``(functions, row_index, function_rows)``.

    Structural identity IS program equality (there is no library-level dedup
    helper, and a re-decode of an unchanged tree yields an equal program tuple), so
    duplicates and elites -- the common case -- merge into one function.

    ``per_row=True`` additionally keys on the row's own ``x`` bytes: tree ``i`` must
    see row ``i``, so two rows that share a program but not their input row are
    genuinely different functions (sharing them would serve wrong numbers).
    Constant trees are keyed by their ``const_out`` alone -- their answer does not
    depend on ``x`` at all.
    """
    functions: List[Any] = []
    positions: Dict[Any, int] = {}
    row_index: List[int] = []
    function_rows: List[int] = []
    for row, tree in enumerate(pop.trees):
        if tree.program is not None:
            key: Any = tree.program
            if per_row:
                key = (tree.program, x_host[row].tobytes())
        else:
            key = ("const", tree.const_out)
        index = positions.get(key)
        if index is None:
            index = len(functions)
            positions[key] = index
            functions.append((tree.program, tree.const_out))
            function_rows.append(row)
        row_index.append(index)
    return tuple(functions), tuple(row_index), tuple(function_rows)


def _trace(fn: Any, specs: Sequence[Any], what: str) -> Any:
    """``evoxc.trace`` with the two structural exception types converted."""
    try:
        return evoxc.trace(fn, *specs)
    except (evoxc.TraceError, evoxc.TransformError) as exc:
        raise BatchUnsupported(
            f"batch_engine: tracing the {what} graph failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _merge(graphs: Sequence[Any], axes: Any, *, switch_ops: bool,
           bands: Optional[int], alignment: str) -> Any:
    """``evoxc.batch_functions`` with the two structural exception types converted."""
    try:
        merged = evoxc.batch_functions(
            list(graphs),
            axes=axes,
            out_axes=0,
            switch_ops=True if switch_ops else None,
            bands=bands,
            alignment=alignment,
        )
    except (evoxc.TraceError, evoxc.TransformError) as exc:
        raise BatchUnsupported(
            f"batch_engine: merging {len(graphs)} per-function graphs failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _require_batching_rules(merged)
    return merged


def _require_batching_rules(merged: Any) -> None:
    """Decline when the merged graph contains an op without a batching rule.

    The merge itself already uses every rule it needs (and a missing one is
    converted by :func:`_merge`), so this is the explicit guard the route contract
    documents: walking the merged entry block keeps a future emitter honest and
    names the offending op instead of surfacing a ``TransformError`` from a later
    stage.  The ``return`` terminator is skipped (it is not an op with semantics
    and has no rule by design).
    """
    ops = getattr(getattr(getattr(merged, "module", None), "main", None),
                  "entry_block", None)
    if ops is None:  # pragma: no cover - evoxc always exposes the entry block
        raise BatchUnsupported("batch_engine: merged graph has no entry block")
    rule_of = _batching.get_batching_rule
    for op in list(ops.ops):
        name = op.name
        if name == "return":
            continue
        if name == "block_call" or name.startswith("external"):
            raise BatchUnsupported(
                f"batch_engine: the merged graph contains an op without a "
                f"batching rule: {name!r}"
            )
        if rule_of(name) is None:
            raise BatchUnsupported(
                f"batch_engine: the merged graph contains an op without a "
                f"batching rule: {name!r}"
            )


def _plan_host(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x_host: np.ndarray,
    label_host: Optional[np.ndarray],
    *,
    per_row: bool,
    variant: str,
    use_mse: bool,
    band_size: Optional[int],
    bands: Optional[Any],
    switch_ops: Optional[bool],
    alignment: Optional[str],
    backend_name: str,
    device: Any,
) -> BatchPlan:
    """The whole host planning step on already-converted inputs (see plan_batch)."""
    input_len = int(input_len)
    output_len = int(output_len)
    pop = decode_population(
        batch_node_value, batch_node_type, batch_subtree_size,
        input_len, output_len,
    )
    pop_size = int(pop.pop_size)
    per_row = bool(per_row)
    variant = _normalize_variant(variant, per_row)
    alignment = resolve_alignment(alignment)
    switch_ops = resolve_switch_ops(switch_ops)
    n_rows = 1 if per_row else int(x_host.shape[0])
    if variant == LOSS_VARIANT:
        if n_rows <= 0:
            raise BatchUnsupported(
                "batch_engine: sr_fitness needs at least one data point, got 0 rows"
            )
        if label_host is None:
            raise BatchUnsupported("batch_engine: the loss variant needs labels")
    functions, row_index, function_rows = _dedup(pop, x_host, per_row)
    band_count = resolve_band_count(
        len(functions), bands=bands, band_size=band_size, switch_ops=switch_ops
    )
    spec = evoxc.TensorSpec((n_rows, input_len), evoxc.float32)
    label_spec = evoxc.TensorSpec((n_rows, output_len), evoxc.float32)
    graphs: List[Any] = []
    for program, const_out in functions:
        if variant == LOSS_VARIANT:
            fn = _loss_fn(program, const_out, n_rows, bool(use_mse))
            graphs.append(_trace(fn, (spec, label_spec), "loss"))
        else:
            fn = _value_fn(program, const_out, n_rows)
            graphs.append(_trace(fn, (spec,), "value"))
    axes = 0 if per_row else (None,) * (2 if variant == LOSS_VARIANT else 1)
    merged = _merge(
        graphs,
        axes,
        switch_ops=switch_ops,
        bands=None if band_count <= 1 else band_count,
        alignment=alignment,
    )
    report = merged.batch_report.as_dict()
    return BatchPlan(
        input_len=input_len,
        output_len=output_len,
        pop_size=pop_size,
        variant=variant,
        use_mse=bool(use_mse),
        per_row=per_row,
        n_rows=n_rows,
        programs=tuple((tree.program, tree.const_out) for tree in pop.trees),
        functions=functions,
        row_index=row_index,
        function_rows=function_rows,
        band_count=int(report.get("bands", band_count) or 1),
        switch_ops=bool(switch_ops),
        alignment=alignment,
        graphs=tuple(graphs),
        merged=merged,
        specs=tuple(merged.tensor_specs),
        report=report,
        backend_name=backend_name,
        device=device,
    )
