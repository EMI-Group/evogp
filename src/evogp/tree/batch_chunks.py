"""Compile-side chunk orchestration of the opt-in batch route.

A merged batch graph compiles as ONE body, and ``evoxc.lower`` / ``evoxc.compile``
/ ``evoxc.load`` of that body is the route's dominant cold cost (linear in the
merged function count, ~0.04-0.11 s per function on the measured envelope).
Nothing about that work depends on the other chunks, so this module splits the
DEDUPED function list into contiguous chunks and stages the bodies in a thread
pool -- the iree compiler runs as a subprocess and releases the GIL, so the
compiles overlap instead of serializing.

What lives here (and what does not):

* :func:`chunk_sizes` -- the partition arithmetic (``base = n // W``, the
  remainder to the FIRST chunks, so the sizes are non-increasing and every chunk
  holds at least ``n // W`` functions).
* :func:`build_bodies` -- the only place the route's ``lower`` / ``compile`` /
  ``load`` statements appear for the chunked path; it returns the executables in
  submission order plus the ``build_stage_ms`` record and folds the region's wall
  clock / job count into :mod:`evogp.tree.batch_stages`.
* NOT here: the front tables, the re-binding rules, the executable key and the
  run path -- those are :mod:`evogp.tree.batch_engine`.  This module never
  imports ``batch_engine``, so the facade stays the only public entry point.

Host work (trace and merge) deliberately stays in the calling thread: the merge
transform is an in-process ``evoxc`` pass whose thread-safety is not documented,
while ``compile`` is a subprocess and ``load`` is a short HAL call per body.
Chunk NUMERICS are unaffected by the split -- each chunk's merged graph computes
its own functions exactly as the single merged graph does, and the caller writes
each chunk's rows into the global ``(n_functions, ...)`` block at the chunk's
offset (see ``batch_engine._run``).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import evoxc

from . import batch_stages
from .evoxc_engine import _gpu_device_kind

__all__ = ["build_bodies", "chunk_sizes"]


def chunk_sizes(n_functions: int, workers: int) -> Tuple[int, ...]:
    """Split ``n_functions`` into ``workers`` contiguous, near-equal chunks.

    ``base = n // workers`` and the ``n % workers`` remainder goes to the FIRST
    chunks, so the sizes are non-increasing and differ by at most one; the caller
    keeps ``workers <= n`` (chunking is only chosen when
    ``n >= MIN_CHUNK_FUNCTIONS``).  An empty function list answers ``()``.
    """
    total = int(n_functions)
    count = int(workers)
    if total < 0:
        raise ValueError(f"n_functions={total} must be >= 0")
    if count < 1:
        raise ValueError(f"workers={count} must be >= 1")
    if total == 0:
        return ()
    count = min(count, total)
    base, remainder = divmod(total, count)
    return tuple([base + 1] * remainder + [base] * (count - remainder))


def _build_body(
    merged: Any,
    *,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
    timer: batch_stages.StageTimer,
) -> Any:
    """The explicit staging sequence of ONE merged graph (lower -> compile -> load).

    ``target_backends`` is the GPU HAL kind read from torch (:func:`_gpu_device_kind`)
    on iree and omitted on numpy; the compile options carry ``opt_level`` only --
    exactly the statements ``build_cached`` / ``evoxc.build`` cannot do for a
    ``Graph``.  A backend failure is propagated, never converted (there is nothing
    safer to fall back to).
    """
    with timer.stage("lower"):
        lowered = evoxc.lower(merged, backend=backend_name)
    with timer.stage("compile"):
        if backend_name == "iree":
            artifact = evoxc.compile(
                lowered,
                backend=backend_name,
                target_backends=[_gpu_device_kind()],
                **options,
            )
        else:
            artifact = evoxc.compile(lowered, backend=backend_name)
    with timer.stage("load"):
        return evoxc.load(artifact, backend=backend_name, device=device)


def build_bodies(
    merges: Sequence[Any],
    indices: Sequence[int],
    *,
    backend_name: str,
    device: Any,
    options: Optional[Dict[str, Any]] = None,
    workers: int = 1,
) -> Tuple[List[Any], Dict[str, float]]:
    """Build every merged graph in ``merges``; returns ``(executables, stage_ms)``.

    ``indices`` labels each body with the chunk index it belongs to (``0`` for the
    single-body plan) and only names the ``build_stage_ms`` keys.  With
    ``workers > 1`` and more than one body the builds run in a
    ``ThreadPoolExecutor``; every job builds its OWN
    :class:`batch_stages.StageTimer` and folds it once under the module lock, so the
    per-stage totals stay correct without a lock per measurement.  The returned
    ``stage_ms`` carries the ``lower`` / ``compile`` / ``load`` totals of this call
    plus ``lower#<index>`` / ``compile#<index>`` / ``load#<index>`` per body.

    A body failure is collected and the FIRST one re-raised (a backend failure is
    not a :class:`evogp.tree.batch_knobs.BatchUnsupported` decline: the remaining
    futures are still collected so no worker leaks, then the exception propagates).
    The region's wall clock and body count are folded into
    ``batch_compile_wall_ms`` / ``batch_compile_jobs`` regardless.
    """
    bodies = list(merges)
    labels = [int(index) for index in indices]
    stage_ms: Dict[str, float] = {}
    if not bodies:
        return [], stage_ms
    executables: List[Any] = [None] * len(bodies)
    locals_ms: List[Dict[str, float]] = [{} for _ in bodies]
    opts = dict(options or {})

    def build_one(slot: int) -> None:
        timer = batch_stages.StageTimer()
        try:
            executables[slot] = _build_body(
                bodies[slot],
                backend_name=backend_name,
                device=device,
                options=opts,
                timer=timer,
            )
        finally:
            locals_ms[slot] = timer.stage_ms
            timer.fold()

    started = time.perf_counter()
    try:
        if int(workers) > 1 and len(bodies) > 1:
            with ThreadPoolExecutor(
                max_workers=min(int(workers), len(bodies)),
                thread_name_prefix="evogp-batch",
            ) as pool:
                futures = [pool.submit(build_one, slot) for slot in range(len(bodies))]
                failures: List[BaseException] = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:  # collected, first one re-raised
                        failures.append(exc)
                if failures:
                    raise failures[0]
        else:
            for slot in range(len(bodies)):
                build_one(slot)
    finally:
        batch_stages.record_compile(
            (time.perf_counter() - started) * 1000.0, len(bodies)
        )
    for name in batch_stages.COMPILE_STAGES:
        stage_ms[name] = 0.0
    for label, local in zip(labels, locals_ms):
        for name in batch_stages.COMPILE_STAGES:
            elapsed = float(local.get(name, 0.0))
            stage_ms[f"{name}#{label}"] = elapsed
            stage_ms[name] += elapsed
    return executables, stage_ms
