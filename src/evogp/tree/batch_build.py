"""The batch route's BUILD path: compile (or reuse) a plan's merged bodies.

:func:`evogp.tree.batch_engine.build_batch` is the route's only build site, and it
is one responsibility: turn a :class:`evogp.tree.batch_programs.BatchPlan` into a
:class:`evogp.tree.batch_programs.BatchExecutable`, reusing whatever already exists
before it compiles anything.  It lives here rather than in the facade so the facade
stays a routing/description surface (the plan memo, ``plan_batch``, the evaluate and
loss entry points, the counters).

Three reuse layers, in order, per body:

1. :data:`evogp.tree.batch_keys.BATCH_EXEC_MEMO` -- the in-process executable table,
   keyed on the actual merged program multiset and the compile options (so a
   re-spelled ``dispatch_formation`` is a different entry);
2. the opt-in disk tier, :mod:`evogp.tree.batch_artifacts` -- one file per compiled
   body under ``EVOGP_EVOXC_ARTIFACT_DIR``, looked up BEFORE anything is lowered; a
   hit skips ``evoxc.lower`` / ``evoxc.compile`` / ``evoxc.load`` entirely;
3. the compile itself -- ``evoxc.lower`` -> ``evoxc.compile`` -> ``evoxc.load`` (the
   explicit sequence, because ``build_cached`` / ``evoxc.build`` reject a ``Graph``),
   through :func:`evogp.tree.batch_chunks.build_bodies` so the chunked plan's bodies
   are staged in a thread pool.

:func:`_build_options` is the plan's options dict: the caller's options plus
``dispatch_formation`` when the plan resolved to ``aggressive`` (the iree adapter
turns it into ``--iree-dispatch-creation-enable-aggressive-fusion``); ``default``
adds no key at all, so the pre-knob build is byte-identical.

Import direction: this module imports the route's leaf modules (``batch_artifacts``,
``batch_chunks``, ``batch_keys``, ``batch_knobs``, ``batch_programs``) and the
engine's backend resolver -- never ``batch_engine``, so the facade can import it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import batch_artifacts, batch_chunks
from .batch_keys import (
    BATCH_EXEC_MEMO,
    _check_executable_matches,
    _chunk_executable_key,
    _executable_key,
)
from .batch_knobs import DISPATCH_FORMATION_AGGRESSIVE, BatchUnsupported
from .batch_programs import BatchChunk, BatchExecutable, BatchPlan
from .evoxc_engine import BACKEND_ENV, SUPPORTED_BACKENDS, _resolve_backend

__all__ = ["build_batch"]


def _resolve_route_backend(
    backend: Optional[str], iree_device_index: Optional[int]
) -> Tuple[str, Any, Dict[str, Any]]:
    """Resolve ``(backend_name, device, options)``; a bad NAME is a decline.

    The name is checked before delegating to the engine's resolver, so an unknown
    backend -- which is a "this route does not build that" answer -- surfaces as
    :class:`BatchUnsupported` instead of the resolver's ``ValueError``.  Every
    other resolver error (an unusable ``EVOGP_EVOXC_OPT_LEVEL``, a negative device
    index) is a configuration mistake and keeps propagating as ``ValueError``.
    """
    name = backend if backend else os.environ.get(BACKEND_ENV, "").strip()
    if name and name not in SUPPORTED_BACKENDS:
        raise BatchUnsupported(
            f"batch_engine: unsupported backend {name!r} (this route builds "
            f"{list(SUPPORTED_BACKENDS)})"
        )
    return _resolve_backend(backend, iree_device_index)



def _build_options(plan: BatchPlan, options: Dict[str, Any]) -> Dict[str, Any]:
    """The compile options of this plan's bodies (``dispatch_formation`` included).

    The resolved spelling on the plan is authoritative: ``"aggressive"`` adds the
    option (the iree adapter translates it into
    ``--iree-dispatch-creation-enable-aggressive-fusion``), ``"default"`` leaves the
    options untouched -- byte-identical to the pre-knob build, and the only value the
    route ever passes to numpy.  The returned dict is a COPY so the caller's
    ``options`` (and every memo key derived from it) stays per-call immutable, and it
    reaches every body of a chunked plan: they share this one dict.
    """
    if plan.dispatch_formation != DISPATCH_FORMATION_AGGRESSIVE:
        return dict(options)
    return {**options, "dispatch_formation": DISPATCH_FORMATION_AGGRESSIVE}


def build_batch(
    plan: BatchPlan,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> BatchExecutable:
    """Compile (or reuse) the merged graph of ``plan``; the ONLY build site.

    ``backend`` / ``iree_device_index`` default to the values the plan was resolved
    with, so ``build_batch(plan_batch(...))`` works without repeating them.

    Contract of :data:`BATCH_EXEC_MEMO` (the only table this function reads): a body
    is keyed by the merged function multiset and shapes -- a HIT is REBOUND to the
    current population's plan, never returned with another population's row
    mapping.  Only the compiled body (and the backend / device it was built for)
    crosses between populations; the returned :class:`BatchExecutable` always
    carries ``plan``, so ``pop_size``, ``row_index``, ``function_rows``, ``programs``
    and the staged specs are this call's.  A repeated evaluation of an unchanged
    population therefore still does no re-lower, re-compile or re-load, and a
    same-multiset population of a different size shares the compile WITHOUT sharing
    its staging.

    Two shapes of build, decided by the PLAN (:func:`_plan_host`), never here
    (``len(plan.chunks)`` is the chunk count in both shapes):

    * ``len(plan.chunks) == 1`` -- the original single body: one ``lower``/
      ``compile``/``load`` of ``plan.merged`` (which is that chunk's graph), stored
      in :data:`BATCH_EXEC_MEMO` as the whole :class:`BatchExecutable`, and
      ``executable`` / ``executables`` both set.
    * ``len(plan.chunks) > 1`` -- one body per chunk: every body not already in
      :data:`BATCH_EXEC_MEMO` (keyed with the partition triple, see
      :func:`_chunk_executable_key`) is built by
      :func:`evogp.tree.batch_chunks.build_bodies` in a thread pool of
      ``plan.compile_workers``; each built body is stored raw and every memo hit is
      re-bound to the caller's chunk after :func:`_check_executable_matches`.  Such
      a plan's ``executable`` is ``None`` -- ``executables`` (one body per chunk) is
      authoritative.

    The build path is the explicit evoxc staging sequence --
    ``evoxc.lower`` -> ``evoxc.compile`` -> ``evoxc.load`` -- because
    ``build_cached`` / ``evoxc.build`` reject a ``Graph``.  ``target_backends`` is
    the GPU HAL kind read from torch (:func:`evogp.tree.evoxc_engine._gpu_device_kind`)
    on iree and omitted on numpy; the compile options carry ``opt_level`` plus
    ``dispatch_formation`` when the plan resolved to ``"aggressive"``
    (:func:`_build_options`), so the flag reaches EVERY body -- the chunked parallel
    compile included, since all bodies share one options dict.

    Optionally, and only when ``EVOGP_EVOXC_ARTIFACT_DIR`` is set, each body is
    looked up on disk BEFORE it is lowered (:mod:`evogp.tree.batch_artifacts`): a hit
    skips ``lower`` / ``compile`` / ``load`` entirely and is re-checked with
    :func:`_check_executable_matches`, a miss compiles just that body and saves it
    afterwards, and a partial hit compiles only the missing chunks.  The tier is off
    by default (one environment read per body, no stat, no key), and a disk hit
    counts as an executable HIT -- it is not a compile miss.

    Raises:
        BatchUnsupported: the resolved backend is not one this route builds, or a
            memoized executable does not fit ``plan``
            (:func:`_check_executable_matches`).
        evoxc errors from ``lower`` / ``compile`` / ``load``: propagated (a backend
            failure is not a decline -- there is nothing safer to fall back to);
            on the chunked path the FIRST such failure is re-raised.
    """
    backend_name, device, options = _resolve_route_backend(
        backend if backend is not None else plan.backend_name,
        iree_device_index,
    )
    options = _build_options(plan, options)
    if len(plan.chunks) <= 1:
        key = _executable_key(plan, backend_name, device, options)
        cached = BATCH_EXEC_MEMO.probe(key)
        if cached is not None:
            BATCH_EXEC_MEMO.record_hit(key)
            # RE-BIND: the compiled body is shared, the caller's plan is not.  Only
            # ``executable`` / ``backend`` / ``device`` may travel across
            # populations (and ``backend``/``device`` are equal by key); the
            # returned value is a FRESH BatchExecutable around the CURRENT plan, so
            # no downstream read of ``.plan`` can see the first population's row
            # mapping.  ``pins`` stays empty: pins belong to the plan-memo entry
            # that holds the population storages (``_plan_entry`` attaches this
            # call's pins), and a cross-population pin would only keep a foreign
            # buffer alive.
            _check_executable_matches(plan, cached.executable)
            return BatchExecutable(
                plan=plan,
                executable=cached.executable,
                executables=(cached.executable,),
                backend=backend_name,
                device=device,
            )
        loaded, path, load_ms = batch_artifacts.load_body(
            plan, plan.chunks[0], backend_name, device, options
        )
        if loaded is not None:
            _check_executable_matches(plan, loaded)
            BATCH_EXEC_MEMO.record_hit(key)
            built = BatchExecutable(
                plan=plan,
                executable=loaded,
                executables=(loaded,),
                backend=backend_name,
                device=device,
                build_stage_ms={"lower": 0.0, "compile": 0.0, "load": load_ms},
            )
            BATCH_EXEC_MEMO.store(key, built)
            return built
        BATCH_EXEC_MEMO.record_miss()
        executables, build_stage_ms = batch_chunks.build_bodies(
            [plan.merged],
            [0],
            backend_name=backend_name,
            device=device,
            options=options,
            workers=1,
        )
        executable = executables[0]
        batch_artifacts.save_body(executable, path)
        built = BatchExecutable(
            plan=plan,
            executable=executable,
            executables=(executable,),
            backend=backend_name,
            device=device,
            build_stage_ms=build_stage_ms,
        )
        BATCH_EXEC_MEMO.store(key, built)
        return built
    bodies: List[Any] = [None] * len(plan.chunks)
    pending: List[Any] = []
    pending_keys: List[Any] = []
    pending_index: List[int] = []
    pending_paths: List[Optional[str]] = []
    disk_ms: Dict[str, float] = {}
    for chunk in plan.chunks:
        key = _chunk_executable_key(plan, chunk, backend_name, device, options)
        cached = BATCH_EXEC_MEMO.probe(key)
        if cached is not None:
            BATCH_EXEC_MEMO.record_hit(key)
            _check_executable_matches(
                plan, cached, specs=chunk.specs, n_functions=len(chunk.functions)
            )
            bodies[chunk.index] = cached
            continue
        loaded, path, load_ms = batch_artifacts.load_body(
            plan, chunk, backend_name, device, options
        )
        if loaded is not None:
            _check_executable_matches(
                plan, loaded, specs=chunk.specs, n_functions=len(chunk.functions)
            )
            BATCH_EXEC_MEMO.record_hit(key)
            BATCH_EXEC_MEMO.store(key, loaded)
            bodies[chunk.index] = loaded
            disk_ms[f"load#{int(chunk.index)}"] = load_ms
            continue
        pending.append(chunk.merged)
        pending_keys.append(key)
        pending_index.append(int(chunk.index))
        pending_paths.append(path)
    if pending:
        built_bodies, build_stage_ms = batch_chunks.build_bodies(
            pending,
            pending_index,
            backend_name=backend_name,
            device=device,
            options=options,
            workers=int(plan.compile_workers),
        )
        for index, key, body, path in zip(
            pending_index, pending_keys, built_bodies, pending_paths
        ):
            BATCH_EXEC_MEMO.record_miss()
            BATCH_EXEC_MEMO.store(key, body)
            batch_artifacts.save_body(body, path)
            bodies[index] = body
    else:
        build_stage_ms = {"lower": 0.0, "compile": 0.0, "load": 0.0}
    build_stage_ms.update(disk_ms)
    return BatchExecutable(
        plan=plan,
        executable=None,
        executables=tuple(bodies),
        backend=backend_name,
        device=device,
        build_stage_ms=build_stage_ms,
    )


