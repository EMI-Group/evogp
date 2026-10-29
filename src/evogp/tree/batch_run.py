"""Run path of the batch route: stage the inputs, run one merged graph per body.

This module owns ONE responsibility for :mod:`evogp.tree.batch_engine` (the
opt-in merged-graph "batch" route): everything that happens AFTER a plan and its
executable exist.  It is deliberately tiny and torch-free on the hot path -- the
merged graph does the arithmetic, so the host does three things per call:

* :func:`_input_block` -- the host copy of the caller's ``x`` (and
  :func:`evogp.tree.batch_programs._host_labels` for the loss variant, which the
  facade converts);
* :func:`_inputs` -- the per-BODY staging: ``per_row=True`` gathers each merged
  function its own ``(1, input_len)`` row, every other layout shares one
  ``(n_rows, input_len)`` block, and a device backend uploads each block once
  (``evoxc.from_numpy`` is a zero-copy host wrap; the upload is
  ``Tensor.to(device)``) while numpy consumes the host array directly;
* :func:`_run` -> :func:`_readback` -> :func:`_map_rows` -> :func:`_to_tensor` --
  one ``evoxc.run`` per compiled body, the single device->host crossing, the host
  row expansion back to population order and the fresh output tensor.

Two shapes of run, decided by the plan's chunk count (``len(plan.chunks)`` is the
chunk count in BOTH plan shapes):

* ``len(plan.chunks) == 1`` -- one staged input tuple, one ``evoxc.run`` of
  :attr:`BatchExecutable.executable`, one readback, returned as-is (the original
  statement of this route);
* ``len(plan.chunks) > 1`` -- the chunk bodies run in ASCENDING chunk order and
  each result is written into a preallocated ``(n_functions, ...)`` block at the
  chunk's ``offset``, so the rows come back in the GLOBAL function order that
  ``plan.row_index`` indexes (the readback order never decides the output order).
  The runs stay SEQUENTIAL: there is one device and one stream, each readback must
  stay ordered with the run that produced it, and a chunk's input staging is a host
  gather -- a pool here would add contention, not throughput.

Stage accounting lives in :mod:`evogp.tree.batch_stages`: ``upload`` wraps the
staging, ``run`` the ``evoxc.run`` and ``gather`` the readback (the facade wraps
the row expansion in ``gather`` as well).  Timing is measurement only: no stage is
reordered and no number depends on it.

Import direction: this module imports :mod:`evogp.tree.batch_knobs`,
:mod:`evogp.tree.batch_programs`, :mod:`evogp.tree.batch_stages` and
:mod:`evogp.tree.evoxc_engine`; it never imports
:mod:`evogp.tree.batch_engine` (the facade imports this module).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc

from . import batch_stages
from .batch_knobs import LOSS_VARIANT, BatchUnsupported
from .batch_programs import BatchExecutable, BatchPlan, _host_rows
from .evoxc_engine import _CPU, _quiet

__all__ = [
    "_input_block",
    "_inputs",
    "_map_rows",
    "_readback",
    "_run",
    "_to_tensor",
]


def _inputs(entry: BatchExecutable, x_host: np.ndarray,
            label_host: Optional[np.ndarray],
            function_rows: Optional[Sequence[int]] = None) -> Tuple[Any, ...]:
    """The graph inputs of one BODY, staged for the entry's backend.

    ``per_row=True`` feeds each merged function its own ``x`` row, so the host
    block is gathered to ``(n_functions, 1, input_len)`` (axes=0 on the function
    axis); every other layout shares one ``(n_rows, input_len)`` block.
    ``function_rows`` is the row slice of the body being fed: ``None`` means the
    whole plan (the single-body path), while a :class:`BatchChunk` passes its own
    ``function_rows`` so a chunked plan gathers exactly the rows its functions were
    first seen at.  On iree each block is uploaded once per call
    (``evoxc.from_numpy`` is a zero-copy host wrap; the upload is
    ``Tensor.to(device)``); the numpy backend consumes the host array directly and
    never uploads.
    """
    plan = entry.plan
    if plan.per_row:
        rows = plan.function_rows if function_rows is None else function_rows
        block = np.ascontiguousarray(
            x_host[list(rows)][:, None, :], dtype=np.float32
        )
    else:
        block = np.ascontiguousarray(x_host, dtype=np.float32)
    staged: List[Any] = [block]
    if label_host is not None:
        staged.append(np.ascontiguousarray(label_host, dtype=np.float32))
    if entry.backend == "numpy":
        return tuple(staged)
    return tuple(evoxc.from_numpy(array).to(entry.device) for array in staged)


def _readback(result: Any, backend: str) -> np.ndarray:
    """Host float32 copy of one body's result (the only device->host crossing).

    ``numpy`` reads the interpreter's array directly; every device backend needs
    ``.to(_CPU)`` + ``.numpy()`` (evoxc has no zero-copy device export).  Both paths
    copy into a fresh ``float32`` array, so a chunk's rows can be written into the
    global block without aliasing a backend buffer.
    """
    if backend == "numpy":
        return np.asarray(result.numpy(), dtype=np.float32)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32)


def _run(entry: BatchExecutable, x_host: np.ndarray,
         label_host: Optional[np.ndarray]) -> np.ndarray:
    """One ``evoxc.run`` per compiled body; the raw host result in FUNCTION order.

    A single-chunk plan (``len(plan.chunks) == 1``) is the original statement: one
    staged input tuple, one ``evoxc.run`` of :attr:`BatchExecutable.executable`, one
    readback, returned as-is.

    A chunked plan (``len(plan.chunks) > 1``) runs its bodies in ASCENDING chunk
    order and writes each result
    into a preallocated ``(n_functions, ...)`` block at the chunk's ``offset``, so
    the rows come back in the GLOBAL function order that ``plan.row_index`` indexes
    (the readback order never decides the output order).  The runs stay SEQUENTIAL:
    there is one device and one stream, each readback must stay ordered with the run
    that produced it, and a chunk's input staging is a host gather -- a pool here
    would add contention, not throughput.

    Stages: ``upload`` wraps the staging, ``run`` the ``evoxc.run`` and ``gather``
    the readback (the caller stages ``_map_rows`` / the output tensor).
    """
    plan = entry.plan
    if len(plan.chunks) <= 1:
        with batch_stages.stage("upload"):
            staged = _inputs(entry, x_host, label_host)
        with batch_stages.stage("run"):
            with _quiet():
                result = evoxc.run(entry.executable, *staged)
        with batch_stages.stage("gather"):
            return _readback(result, entry.backend)
    tail: Tuple[int, ...] = (
        () if plan.variant == LOSS_VARIANT else (int(plan.n_rows), int(plan.output_len))
    )
    rows = np.empty((len(plan.functions),) + tail, dtype=np.float32)
    executables = tuple(entry.executables)
    if len(executables) != len(plan.chunks):
        raise BatchUnsupported(
            f"batch_engine: the chunked plan holds {len(plan.chunks)} chunk(s) but "
            f"{len(executables)} executable(s) were bound to it"
        )
    for chunk, executable in zip(plan.chunks, executables):
        size = len(chunk.functions)
        with batch_stages.stage("upload"):
            staged = _inputs(entry, x_host, label_host, chunk.function_rows)
        with batch_stages.stage("run"):
            with _quiet():
                result = evoxc.run(executable, *staged)
        with batch_stages.stage("gather"):
            block = _readback(result, entry.backend)
            if block.shape[0] != size:
                raise BatchUnsupported(
                    f"batch_engine: chunk {chunk.index} was expected to answer "
                    f"{size} function(s), got {block.shape[0]}"
                )
            rows[chunk.offset:chunk.offset + size] = block
    return rows


def _map_rows(rows: np.ndarray, plan: BatchPlan) -> np.ndarray:
    """Expand the per-function rows back to the population order (a fresh copy)."""
    return np.ascontiguousarray(rows[list(plan.row_index)])


def _to_tensor(array: np.ndarray, source: Any, out_device: Optional[Any]) -> torch.Tensor:
    """A fresh, contiguous, ``requires_grad=False`` float32 tensor on the target."""
    tensor = torch.from_numpy(array).contiguous().detach().requires_grad_(False)
    target = out_device if out_device is not None else getattr(source, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


def _input_block(entry: BatchExecutable, x: Any, input_len: int,
                 per_row: bool) -> np.ndarray:
    """Host copy of ``x`` for the run.

    Deliberately re-read rather than memoized: the two front tables are keyed on
    the POPULATION (a plan) and on the compiled body (an executable), and a
    per-call ``x`` conversion is one D2H -- the same single crossing the dense
    route's input table would save on a repeated dataset, without a third table to
    keep coherent (the wiring round can add one if it measures a need).
    """
    return _host_rows(x, int(input_len), "x", per_row=bool(per_row),
                      pop_size=entry.plan.pop_size)
