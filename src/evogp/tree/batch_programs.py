"""Host-side program handling of the batch route: decode, dedup, trace, merge.

Everything between a population's three tensors and its merged
:class:`evoxc.Graph` objects lives here: the input / label conversions, the
per-function graph bodies, the structural dedup, the ``evoxc.trace`` /
``evoxc.batch_functions`` calls with their two structural exception types converted
into :class:`evogp.tree.batch_knobs.BatchUnsupported`, :func:`plan_digest` (the
content key of a plan), and :func:`_plan_host`, which produces the
:class:`BatchPlan` of one call (the frozen programs, the row mapping, the traced
graphs, and one :class:`BatchChunk` per compiled body -- the single-body plan's one
chunk mirrors its whole-graph merged graph, a chunked plan's chunks are contiguous
slices of the deduped functions).

No compile, no front table and no torch tensor assembly happen here -- that is
:mod:`evogp.tree.batch_engine` (front tables, run path) and
:mod:`evogp.tree.batch_chunks` (the ``lower`` / ``compile`` / ``load`` pool).  This
module never imports ``batch_engine`` back at module level, so
``evogp.tree.batch_engine`` stays the public facade.

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

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc
from evoxc.transforms import batching as _batching

# ``batch_knobs`` is imported as a MODULE too: ``MIN_CHUNK_FUNCTIONS`` is read
# through the module object at call time (never captured by value), so a test can
# monkeypatch it to force chunking on a tiny fixture.
from . import batch_bands, batch_chunks, batch_knobs, batch_stages
from .batch_chunks import chunk_sizes
from .batch_knobs import (
    DISPATCH_FORMATION_AUTO,
    DISPATCH_FORMATION_DEFAULT,
    LOSS_VARIANT,
    BatchUnsupported,
    _normalize_variant,
    effective_dispatch_formation,
    resolve_alignment,
    resolve_band_size,
    resolve_bands,
    resolve_compile_workers,
    resolve_dispatch_formation,
    resolve_switch_ops,
)
from .evoxc_engine import (
    _bits_to_float,
    _trace_const_col,
    _walk_tokens,
    decode_population,
)

#: Version tag of :func:`plan_digest`.  Bump it whenever the encoded content or its
#: order changes: an old digest can then never match a new plan (the content table
#: stores plans, and serving a stale one would hand out another population's rows).
PROGRAM_DIGEST_FORMAT = "evogp-batch-plan-digest-v1"


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
class BatchChunk:
    """One compiled body of a plan (a contiguous slice of the deduped functions).

    EVERY plan carries one chunk per compiled body (:attr:`BatchPlan.chunks`), so
    the record is always the authoritative read surface: a single-chunk plan's
    ``chunks[0]`` covers the whole function list and its ``graphs`` / ``merged`` /
    ``specs`` / ``report`` ARE the plan's whole-graph fields (same objects), while a
    chunked plan's chunks are proper contiguous slices and the plan's whole-graph
    fields stay empty.

    Attributes:
        index: the chunk's position (``0 <= index < chunk_count``); it is also the
            key label of the ``build_stage_ms`` entries and the slot of the
            executable in :attr:`BatchExecutable.executables`.
        offset: the chunk's first index in the plan's GLOBAL ``functions`` /
            ``function_rows`` tuples, i.e. where its rows are written back into the
            ``(n_functions, ...)`` block; ``sum(previous sizes)`` by construction, so
            the global output order is the function order (never the chunk order).
        functions: this chunk's slice of the deduped ``(program, const_out)``
            multiset (in global order).
        function_rows: the matching slice of ``BatchPlan.function_rows`` (the
            population row each function was first seen at, i.e. the ``x`` row a
            ``per_row`` plan feeds it).
        graphs: the traced per-function :class:`evoxc.Graph` objects of this chunk.
        merged: this chunk's merged :class:`evoxc.Graph` (the plan's own graph for a
            single-chunk plan).
        specs: the chunk's merged tensor input specs (``evoxc.TensorSpec``).
        report: ``merged.batch_report.as_dict()`` of this chunk, plus the five
            band-policy keys (``band_requested`` / ``band_requested_bands`` /
            ``band_clamped`` / ``band_note`` / ``band_policy``) of the chunk's own
            :class:`evogp.tree.batch_bands.BandSpec` -- so a chunked plan's
            authoritative band story is per chunk.
        band_count: the chunk's own band count, resolved from the plan's band
            request sized to the chunk
            (:func:`evogp.tree.batch_bands.chunk_band_spec`) and clamped by the
            device's binding policy; ``report["bands"]`` is the same number.
    """

    index: int
    offset: int
    functions: Tuple[Any, ...]
    function_rows: Tuple[int, ...]
    graphs: Tuple[Any, ...]
    merged: Any
    specs: Tuple[Any, ...]
    report: Dict[str, Any]
    band_count: int


@dataclass(frozen=True)
class BatchPlan:
    """The host-side plan of one population (one merged graph, or N chunk bodies).

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
            This is the GLOBAL function order for both plan shapes: a chunked plan's
            :attr:`chunks` are contiguous slices of it and their rows are assembled
            back into this order.
        row_index: per population row, its (GLOBAL) index into :attr:`functions`.
        function_rows: per merged function, the population row it was first seen
            at (the ``x`` row a ``per_row`` plan feeds it).
        band_count: number of merge bands (``1`` = the flat merge).  A chunked plan
            always reports ``1``; the per-chunk band counts live in
            :attr:`BatchChunk.band_count`.
        switch_ops: whether op-switching was requested (forces ``band_count == 1``).
        alignment: the ``switch_ops`` class-alignment mode.
        graphs: the traced per-function :class:`evoxc.Graph` objects of the plan --
            populated only when ``len(chunks) == 1`` (``()`` on a chunked plan, whose
            per-chunk graphs live in :attr:`BatchChunk.graphs`).
        merged: the merged :class:`evoxc.Graph` -- populated only when
            ``len(chunks) == 1`` (``None`` on a chunked plan).  ``merged.batch_report``
            describes it, and it IS ``chunks[0].merged``.
        specs: the merged graph's tensor input specs (``evoxc.TensorSpec``) --
            populated only when ``len(chunks) == 1`` (``()`` on a chunked plan), and
            equal to ``chunks[0].specs``.
        report: ``merged.batch_report.as_dict()`` when ``len(chunks) == 1`` (the same
            dict as ``chunks[0].report``); for a chunked plan an honest aggregate
            (``chunk_count`` / ``chunk_sizes`` / ``n_functions`` / the summed
            ``emitted_ops``, with ``bands`` / ``band_sizes`` / ``band_strategy`` =
            ``None``), because no single graph describes the build.
        backend_name: the backend the plan was resolved for (the default of
            :func:`build_batch`).
        device: the resolved ``evoxc.core.Device``.
        stage_ms: the PLAN-side stage totals of the call that BUILT it
            (``decode`` / ``dedup`` / ``trace`` / ``merge``; ``decode`` is absent
            when the plan was built from an already-decoded population).
        compile_workers: the number of compiled bodies this plan holds, i.e. the
            worker count ACTUALLY used (``1`` for the single-body path, including
            ``EVOGP_BATCH_COMPILE_WORKERS=0``).
        chunk_sizes: per compiled body, its function count (``(len(functions),)``
            for the single-body path); equal to
            ``tuple(len(chunk.functions) for chunk in chunks)``.
        chunks: one :class:`BatchChunk` per compiled body -- ALWAYS populated
            (``len(chunks) >= 1``) and the authoritative read surface of the plan,
            with ``chunk_count == len(chunks) >= 1``.
        band_requested: the band request spelling the plan resolved
            (``"flat"`` / ``"explicit"`` / ``"size"`` / ``"auto"``, i.e.
            ``BandSpec.requested``) -- the single-body spec's, or the shared one of
            a chunked plan.
        band_clamped: whether the binding policy lowered the band count -- the
            single-body spec's clamp (plus the ``auto`` re-merge), or ``True`` when
            ANY chunk of a chunked plan was clamped.
        band_note: the clamp explanation, else ``None`` (a chunked plan joins the
            clamped chunks' notes with ``" | "``).
        band_policy: the resolved :class:`evogp.tree.batch_bands.BandPolicy` as a
            dict (``backend`` / ``device_kind`` / ``max_bands`` / ``min_band_size``
            / ``uncapped``).
        dispatch_formation: the EFFECTIVE
            ``EVOGP_BATCH_DISPATCH_FORMATION`` spelling this plan's merged bodies are
            compiled with -- ``"aggressive"`` or ``"default"``, the value the
            compile options carry and the memo keys include
            (:func:`plan_dispatch_formation`).
        dispatch_formation_requested: the request it came from (``"auto"`` /
            ``"aggressive"`` / ``"default"``), so a record can state whether the
            target policy chose the flag or the caller did.

    :attr:`chunks` is the authoritative read surface for both plan shapes; the
    whole-graph fields (:attr:`graphs` / :attr:`merged` / :attr:`specs` /
    :attr:`report`'s band keys) are populated only when ``len(chunks) == 1``, where
    ``chunks[0]`` mirrors them object-for-object, and ``chunk_sizes`` says how many
    bodies this plan holds in both shapes.
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
    stage_ms: Dict[str, float] = field(default_factory=dict)
    compile_workers: int = 1
    chunk_sizes: Tuple[int, ...] = ()
    chunks: Tuple[BatchChunk, ...] = ()
    band_requested: Optional[str] = None
    band_clamped: bool = False
    band_note: Optional[str] = None
    band_policy: Dict[str, Any] = field(default_factory=dict)
    dispatch_formation: str = DISPATCH_FORMATION_DEFAULT
    dispatch_formation_requested: str = DISPATCH_FORMATION_AUTO


@dataclass(frozen=True)
class BatchExecutable:
    """A plan plus its compiled executable(s) (the value of all three front tables).

    Attributes:
        plan: the :class:`BatchPlan` this build is for.
        executable: the single merged body's loaded ``evoxc.Executable`` --
            ``executables[0]`` when ``len(plan.chunks) == 1`` and ``None`` on a
            chunked plan (then :attr:`executables` is authoritative).
            The single-body path is the default, so this field keeps its original
            meaning; every internal read site goes through :attr:`executables` (or
            ``plan.chunks``) and never assumes one body.
        backend: the backend name it was compiled for.
        device: the ``evoxc.core.Device`` it was loaded on.
        pins: strong references to the population/input storages the plan memo entry
            is keyed on (empty for an unmemorized build); they only keep the
            ``data_ptr`` identity tokens ABA-proof, so they are never inspected.
        build_stage_ms: the compile-side stage totals of the call that BUILT it --
            ``lower`` / ``compile`` / ``load`` (summed over the bodies) plus
            ``lower#<i>`` / ``compile#<i>`` / ``load#<i>`` per chunk index.  All
            zero for a build served entirely from :data:`BATCH_EXEC_MEMO`.
        executables: one ``evoxc.Executable`` per compiled body, in chunk order
            (``(executable,)`` for the single-body path); the run path iterates this
            tuple together with ``plan.chunks`` and writes each chunk's rows at the
            chunk's ``offset``.
    """

    plan: BatchPlan
    executable: Any
    backend: str
    device: Any
    pins: Tuple[Any, ...] = ()
    build_stage_ms: Dict[str, float] = field(default_factory=dict)
    executables: Tuple[Any, ...] = ()


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
           bands: Optional[Any], alignment: str) -> Any:
    """``evoxc.batch_functions`` with the two structural exception types converted.

    ``bands`` is passed through UNCHANGED: ``None`` (the flat merge), an ``int``
    ``K`` (that many contiguous bands) or the string ``"auto"`` (evoxc's cost model
    picks the partition).  Which of the three a call asks for is decided by the
    binding policy in :mod:`evogp.tree.batch_bands`, not here.
    """
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


def plan_digest(
    programs: Sequence[Tuple[Any, Any]],
    x_host: Optional[np.ndarray],
    per_row: bool,
) -> bytes:
    """A strong content digest of a population's decoded programs (32 bytes).

    The plan's CONTENT key: two populations with the same bytes must answer with the
    same plan, however their tensors were allocated.  Fed in POPULATION ORDER (the
    function order IS the merged graph's leading axis, so the order is part of the
    content):

    * a version tag, then the ``per_row`` flag;
    * per row, the row index, then either ``const:<len>|<bits>`` (a fully constant
      tree, ``const_out`` float32 bit patterns) or ``prog:<out_len>:<len>|<tokens>``
      (the frozen static program: ``(kind, value...)`` int tokens);
    * for ``per_row``, that row's own ``x`` row bytes -- the dedup keys on them,
      because tree ``i`` must see input row ``i``, so two rows that share a program
      but not their input row are genuinely different functions.

    ``blake2b(digest_size=32)`` is a MEMBERSHIP test, never the identity: a plan hit
    derived from it is verified against the decoded programs before it is served
    (:func:`evogp.tree.batch_engine._plan_entry`), so a collision costs a rebuild,
    never a wrong number.  Nothing is memoized here; this is pure host work.
    """
    digest = hashlib.blake2b(digest_size=32)
    digest.update(PROGRAM_DIGEST_FORMAT.encode("ascii"))
    digest.update(b"|per_row=%d|" % int(bool(per_row)))
    rows = 0 if x_host is None else int(x_host.shape[0])
    for row, (program, const_out) in enumerate(programs):
        digest.update(b"|row=%d|" % row)
        if program is None:
            digest.update(b"const=%d|" % len(const_out))
            digest.update(_int_bytes(const_out))
        else:
            out_len, tokens = program
            digest.update(b"prog=%d:%d|" % (int(out_len), len(tokens)))
            digest.update(_int_bytes(tokens))
        if per_row and row < rows:
            digest.update(b"x=")
            digest.update(
                np.ascontiguousarray(x_host[row], dtype=np.float32).tobytes()
            )
    return digest.digest()


def _int_bytes(tokens: Any) -> bytes:
    """Little-endian int64 bytes of one program's token/int sequence.

    The tokens are ints by construction (``(kind, value...)``); anything that does
    not convert falls back to ``repr`` so an exotic program can never crash the
    digest (it only costs a slower, still-injective encoding).
    """
    try:
        return np.asarray(tokens, dtype=np.int64).tobytes()
    except (TypeError, ValueError):  # pragma: no cover - tokens are int tuples
        return repr(tokens).encode("utf-8")


def plan_dispatch_formation(
    backend_name: Optional[str], device: Any, value: Optional[str] = None
) -> Tuple[str, str]:
    """``(requested, effective)`` dispatch formation of ONE plan.

    The REQUEST comes from the knob (:func:`evogp.tree.batch_knobs.resolve_dispatch_formation`,
    i.e. ``explicit argument > EVOGP_BATCH_DISPATCH_FORMATION > auto``); the EFFECTIVE
    spelling applies :func:`evogp.tree.batch_knobs.effective_dispatch_formation` with
    the target read off the SAME band policy that resolves this plan's device
    (:func:`evogp.tree.batch_bands.resolve_band_policy`, the one place the compile
    target's device kind is decided), so the GPU-only enable rule can never drift
    from the device the merged graph is compiled for.  The evoxc feature probe
    (:func:`evogp.tree.batch_chunks.supports_dispatch_formation`) is consulted only
    for ``iree`` -- ``numpy`` documents options as ignored, so the key is never put
    there.

    An unusable VALUE raises ``ValueError`` naming the variable; an option the
    installed evoxc does not know degrades to ``default`` instead (see the knob).
    """
    requested = resolve_dispatch_formation(value)
    name = "" if backend_name is None else str(backend_name).strip().lower()
    kind = batch_bands.resolve_band_policy(name or None, device).device_kind
    supported = (
        batch_chunks.supports_dispatch_formation(name) if name == "iree" else False
    )
    effective = effective_dispatch_formation(
        requested,
        backend=name,
        gpu=kind in batch_bands.GPU_DEVICE_KINDS,
        supported=supported,
    )
    return requested, effective


def _resolve_workers(
    requested: Optional[int],
    n_functions: int,
    *,
    bands: Optional[Any],
    switch_ops: bool,
    chunk_bands: Optional[bool] = None,
) -> int:
    """The worker count ACTUALLY used for this plan (the ``W = 1`` forcing rules).

    ``resolve_compile_workers`` answers the request (``explicit argument >
    environment > default``); this clamp then decides whether the plan is chunked:

    * ``W <= 1`` -- the single-body path verbatim (the original one-graph plan);
    * ``n_functions < batch_knobs.MIN_CHUNK_FUNCTIONS`` -- one compile cannot be
      split usefully below the per-body fixed cost, so the single-body path stays;
    * ``switch_ops`` -- op-switching keeps its documented per-plan meaning and is
      never combined with banding, so it always forces the single-body path;
    * a band request (``bands is not None``, which now includes the ``auto``
      spelling, the flat spellings included since ``resolve_bands`` answers ``1``
      for them) -- refused unless ``resolve_chunk_bands(chunk_bands)`` opts into the
      chunk-banded composition, because the band sweep is measured on the
      single-body path;
    * otherwise ``W = min(W, n_functions // MIN_CHUNK_FUNCTIONS)``, so every chunk
      keeps at least ``MIN_CHUNK_FUNCTIONS`` functions and the chunk count equals
      ``W`` exactly.

    ``MIN_CHUNK_FUNCTIONS`` is read through the MODULE object (never captured), so a
    test can lower it to force chunking on a tiny fixture.
    """
    workers = int(resolve_compile_workers(requested))
    if workers <= 1:
        return 1
    minimum = max(1, int(batch_knobs.MIN_CHUNK_FUNCTIONS))
    if int(n_functions) < minimum:
        return 1
    if switch_ops:
        return 1
    if bands is not None and not batch_knobs.resolve_chunk_bands(chunk_bands):
        return 1
    return max(1, min(workers, int(n_functions) // minimum))


def _trace_functions(
    functions: Sequence[Any],
    variant: str,
    use_mse: bool,
    n_rows: int,
    specs: Sequence[Any],
) -> List[Any]:
    """Trace one graph per function, in order (the per-distinct-program loop)."""
    graphs: List[Any] = []
    for program, const_out in functions:
        if variant == LOSS_VARIANT:
            graphs.append(
                _trace(_loss_fn(program, const_out, n_rows, bool(use_mse)), specs, "loss")
            )
        else:
            graphs.append(_trace(_value_fn(program, const_out, n_rows), specs, "value"))
    return graphs


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
    pop: Any = None,
    compile_workers: Optional[int] = None,
    chunk_bands: Optional[bool] = None,
    dispatch_formation: Optional[str] = None,
) -> BatchPlan:
    """The whole host planning step on already-converted inputs (see plan_batch).

    ``pop`` is the ALREADY-decoded population when the caller decoded it anyway
    (:func:`evogp.tree.batch_engine._plan_entry` needs the programs for the plan's
    content key) -- that is the only way the decode is not paid twice; ``None``
    decodes here, which is what :func:`evogp.tree.batch_engine.plan_batch` relies on.

    The band request is resolved ONCE, for the deduped function count, by
    :func:`evogp.tree.batch_bands.resolve_band_plan` (the backend/device binding
    policy included), so ``auto`` / an explicit ``K`` / the band-size model all
    reach the merge through one :class:`evogp.tree.batch_bands.BandSpec`.

    ``compile_workers`` is resolved with ``resolve_compile_workers`` and then
    clamped by :func:`_resolve_workers`: the plan is CHUNKED only when the resolved
    count is ``> 1``, the deduped population has at least
    ``batch_knobs.MIN_CHUNK_FUNCTIONS`` functions, ``switch_ops`` is off and a band
    request is either absent or opted into chunking via
    ``EVOGP_BATCH_CHUNK_BANDS`` (``chunk_bands``, explicit argument first).  A
    chunked plan keeps the GLOBAL function order (``functions`` / ``row_index`` /
    ``function_rows`` are unchanged and each chunk is a contiguous slice with an
    ``offset``), so the assembled output order is exactly the single-body plan's.
    Both shapes store one :class:`BatchChunk` per compiled body in ``chunks`` (the
    single-body plan's one chunk mirrors the whole-graph fields), so
    ``len(plan.chunks)`` is the plan's chunk count.

    ``dispatch_formation`` is the ``EVOGP_BATCH_DISPATCH_FORMATION`` request (the
    environment when ``None``); it is resolved ONCE here with the plan's own
    backend/device by :func:`plan_dispatch_formation`, and the resulting spelling is
    stored on the plan (``dispatch_formation`` / ``dispatch_formation_requested``) so
    the build, the memo keys and ``describe_plan`` all read one value.
    """
    timer = batch_stages.StageTimer()
    input_len = int(input_len)
    output_len = int(output_len)
    if pop is None:
        with timer.stage("decode"):
            pop = decode_population(
                batch_node_value,
                batch_node_type,
                batch_subtree_size,
                input_len,
                output_len,
            )
    pop_size = int(pop.pop_size)
    per_row = bool(per_row)
    variant = _normalize_variant(variant, per_row)
    alignment = resolve_alignment(alignment)
    switch_ops = resolve_switch_ops(switch_ops)
    band_size = resolve_band_size(band_size)
    explicit_bands = resolve_bands(bands)
    formation_requested, formation = plan_dispatch_formation(
        backend_name, device, dispatch_formation
    )
    n_rows = 1 if per_row else int(x_host.shape[0])
    if variant == LOSS_VARIANT:
        if n_rows <= 0:
            raise BatchUnsupported(
                "batch_engine: sr_fitness needs at least one data point, got 0 rows"
            )
        if label_host is None:
            raise BatchUnsupported("batch_engine: the loss variant needs labels")
    with timer.stage("dedup"):
        functions, row_index, function_rows = _dedup(pop, x_host, per_row)
    n_functions = len(functions)
    workers = _resolve_workers(
        compile_workers,
        n_functions,
        bands=explicit_bands,
        switch_ops=switch_ops,
        chunk_bands=chunk_bands,
    )
    band_spec = batch_bands.resolve_band_plan(
        n_functions,
        bands=explicit_bands,
        band_size=band_size,
        switch_ops=switch_ops,
        backend=backend_name,
        device=device,
    )
    spec = evoxc.TensorSpec((n_rows, input_len), evoxc.float32)
    label_spec = evoxc.TensorSpec((n_rows, output_len), evoxc.float32)
    specs: Tuple[Any, ...] = (spec, label_spec) if variant == LOSS_VARIANT else (spec,)
    axes = 0 if per_row else (None,) * (2 if variant == LOSS_VARIANT else 1)
    programs = tuple((tree.program, tree.const_out) for tree in pop.trees)
    if workers <= 1:
        with timer.stage("trace"):
            graphs = _trace_functions(functions, variant, use_mse, n_rows, specs)
        with timer.stage("merge"):
            merged, report = batch_bands.merge_with_spec(
                _merge,
                graphs,
                axes,
                band_spec,
                switch_ops=switch_ops,
                alignment=alignment,
                auto_total=n_functions,
            )
        plan_band_count = int(report.get("bands", band_spec.num_bands or 1) or 1)
        plan_graphs = tuple(graphs)
        plan_specs = tuple(merged.tensor_specs)
        # One chunk per BODY in both plan shapes: the single-chunk chunk mirrors the
        # whole-graph fields (same objects) so ``plan.chunks`` is always the
        # authoritative read surface, while the top-level fields stay populated.
        chunk = BatchChunk(
            index=0,
            offset=0,
            functions=functions,
            function_rows=function_rows,
            graphs=plan_graphs,
            merged=merged,
            specs=plan_specs,
            report=report,
            band_count=plan_band_count,
        )
        plan = BatchPlan(
            input_len=input_len,
            output_len=output_len,
            pop_size=pop_size,
            variant=variant,
            use_mse=bool(use_mse),
            per_row=per_row,
            n_rows=n_rows,
            programs=programs,
            functions=functions,
            row_index=row_index,
            function_rows=function_rows,
            band_count=plan_band_count,
            switch_ops=bool(switch_ops),
            alignment=alignment,
            graphs=plan_graphs,
            merged=merged,
            specs=plan_specs,
            report=report,
            backend_name=backend_name,
            device=device,
            stage_ms=timer.stage_ms,
            compile_workers=1,
            chunk_sizes=(n_functions,),
            chunks=(chunk,),
            band_requested=band_spec.requested,
            band_clamped=bool(report["band_clamped"]),
            band_note=report["band_note"],
            band_policy=report["band_policy"],
            dispatch_formation=formation,
            dispatch_formation_requested=formation_requested,
        )
        timer.fold()
        return plan
    sizes = chunk_sizes(n_functions, workers)
    bounds: List[Tuple[int, int]] = []
    offset = 0
    for size in sizes:
        bounds.append((offset, offset + size))
        offset += size
    with timer.stage("trace"):
        traced: List[List[Any]] = [
            _trace_functions(functions[start:stop], variant, use_mse, n_rows, specs)
            for start, stop in bounds
        ]
    with timer.stage("merge"):
        merged_graphs: List[Any] = []
        reports: List[Dict[str, Any]] = []
        chunk_band_counts: List[int] = []
        for (start, stop), graphs in zip(bounds, traced):
            # Per-chunk band spec: the SAME band request as the plan, sized to this
            # chunk (``auto`` per chunk, the proportional share of an explicit ``K``,
            # the size model re-applied) and vetted by the same binding policy.
            chunk_spec = batch_bands.chunk_band_spec(
                band_spec, stop - start, n_functions, band_size=band_size
            )
            merged_chunk, report = batch_bands.merge_with_spec(
                _merge,
                graphs,
                axes,
                chunk_spec,
                switch_ops=False,
                alignment=alignment,
                auto_total=stop - start,
            )
            merged_graphs.append(merged_chunk)
            reports.append(report)
            chunk_band_counts.append(int(report.get("bands") or 1))
    chunks = tuple(
        BatchChunk(
            index=index,
            offset=start,
            functions=functions[start:stop],
            function_rows=function_rows[start:stop],
            graphs=tuple(graphs),
            merged=merged_graphs[index],
            specs=tuple(merged_graphs[index].tensor_specs),
            report=reports[index],
            band_count=chunk_band_counts[index],
        )
        for index, ((start, stop), graphs) in enumerate(zip(bounds, traced))
    )
    aggregate = {
        "chunk_count": len(chunks),
        "chunk_sizes": [len(chunk.functions) for chunk in chunks],
        "n_functions": n_functions,
        "emitted_ops": sum(
            int((chunk.report or {}).get("emitted_ops") or 0) for chunk in chunks
        ),
        "bands": None,
        "band_sizes": None,
        "band_strategy": None,
        # The plan-level band story of a chunked plan (the request every chunk
        # repeats, the summed auto counts before clamping, any chunk's clamp).
        **batch_bands.aggregate_band_report(band_spec, reports),
    }
    plan = BatchPlan(
        input_len=input_len,
        output_len=output_len,
        pop_size=pop_size,
        variant=variant,
        use_mse=bool(use_mse),
        per_row=per_row,
        n_rows=n_rows,
        programs=programs,
        functions=functions,
        row_index=row_index,
        function_rows=function_rows,
        band_count=1,
        switch_ops=bool(switch_ops),
        alignment=alignment,
        graphs=(),
        merged=None,
        specs=(),
        report=aggregate,
        backend_name=backend_name,
        device=device,
        stage_ms=timer.stage_ms,
        compile_workers=len(chunks),
        chunk_sizes=sizes,
        chunks=chunks,
        band_requested=band_spec.requested,
        band_clamped=bool(aggregate["band_clamped"]),
        band_note=aggregate["band_note"],
        band_policy=aggregate["band_policy"],
        dispatch_formation=formation,
        dispatch_formation_requested=formation_requested,
    )
    timer.fold()
    return plan
