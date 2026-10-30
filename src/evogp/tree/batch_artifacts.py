"""The batch route's tier of the on-disk executable cache (``artifact_cache`` reused).

:mod:`evogp.tree.artifact_cache` already turns one compiled ``evoxc.Executable``
into a file and back, keyed by everything that determines the compiled body.  This
module adds the LITTLE that is route-specific and nothing else:

* **what a batch body is called** -- :func:`body_name` composes the plan's variant,
  the chunk index and a CONTENT digest of that chunk's merged functions
  (:func:`evogp.tree.batch_programs.plan_digest` over the deduped function list --
  the same bytes the executable memo keys on, so two plans with different programs
  can never share a file, and the FUNCTION ORDER is part of it), and
  :func:`body_path` hands that name to :func:`evogp.tree.artifact_cache.artifact_path`
  with the region's own subdirectory and the route's compile options (which carry
  the resolved ``dispatch_formation`` spelling, so a different spelling is a
  different file).
* **the route's own counters** -- ``batch_artifact_*`` (``stats`` /
  ``reset_stats``), disjoint from ``artifact_cache_stats()`` (the whole disk layer,
  shared with the dense and specialized routes), from the eight-key
  ``batch_cache_stats()`` and from ``cache_stats()``: a record can state disk hits
  and misses per call without a shared counter changing meaning.
* :func:`load_body` / :func:`save_body` -- the two calls :func:`evogp.tree.batch_engine.build_batch`
  makes per compiled body: load BEFORE lower/compile/load (a hit skips all three),
  save AFTER a real build (best effort).

The switch, the key, the atomic write and the validation are entirely
:mod:`evogp.tree.artifact_cache`'s: ``EVOGP_EVOXC_ARTIFACT_DIR`` unset/empty means
:func:`body_path` answers ``None`` after one environment read, so
:func:`load_body` returns immediately (no stat, no key, no directory) and
:func:`save_body` does nothing -- exactly the route's pre-cache behaviour.  A
corrupt, truncated or foreign file is discarded by
:func:`evogp.tree.artifact_cache.load` (which validates it with
:func:`evogp.tree.artifact_cache.signature_matches`) and costs one rebuild, never a
wrong number; a plan-level load is additionally re-checked by
:func:`evogp.tree.batch_keys._check_executable_matches` on the way in.

Import direction: this module imports :mod:`evogp.tree.artifact_cache`,
:mod:`evogp.tree.batch_programs` and (for the chunk/plan records) nothing else that
imports it -- the facade and :mod:`evogp.tree.batch_engine` are the only callers.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from . import artifact_cache
from .batch_programs import BatchChunk, BatchPlan, plan_digest

__all__ = [
    "DIR_ENV",
    "SUBDIR",
    "SUFFIX",
    "body_digest",
    "body_name",
    "body_path",
    "load",
    "load_body",
    "reset_stats",
    "save",
    "save_body",
    "stats",
]

#: The switch is the engine's (``EVOGP_EVOXC_ARTIFACT_DIR``): one directory serves
#: every route, and unset/empty turns all of them off.
DIR_ENV = artifact_cache.DIR_ENV
#: Region of the batch route below that directory -- batch artifacts can never
#: collide with the dense/specialized ones (which keep the root), and one
#: ``EVOGP_EVOXC_ARTIFACT_DIR`` serves the whole engine.
SUBDIR = "batch"
SUFFIX = artifact_cache.SUFFIX

#: Characters of the content digest kept in a file name (a readable, collision-safe
#: prefix -- the file's own key is still the full sha256 artifact key).
DIGEST_CHARS = 16


def body_digest(functions: Sequence[Any]) -> str:
    """Content digest of ONE body's merged function list (hex, ``DIGEST_CHARS``).

    :func:`evogp.tree.batch_programs.plan_digest` is the route's one content
    encoding (program tokens, per-row index, order-sensitive), so this digest means
    the same thing in every process and changes whenever the merged functions or
    their order do.
    """
    return plan_digest(functions, None, False).hex()[:DIGEST_CHARS]


def body_name(plan: BatchPlan, chunk: BatchChunk) -> str:
    """Readable graph name of one compiled body: ``batch:<variant>:c<i>:<digest>``.

    The variant and the chunk index keep a plan's bodies apart inside one process,
    the digest pins the merged content, and the artifact key derived from the name
    (plus the specs, the backend, the device and the frozen compile options, the
    resolved ``dispatch_formation`` among them) is what a later process looks up.
    """
    return (
        f"batch:{plan.variant}:c{int(chunk.index)}:{body_digest(chunk.functions)}"
    )


def body_path(
    plan: BatchPlan,
    chunk: BatchChunk,
    backend_name: str,
    device: Any,
    options: Any,
) -> Optional[str]:
    """File path of one compiled body, or ``None`` (switch off / key underivable)."""
    return artifact_cache.artifact_path(
        body_name(plan, chunk),
        tuple(chunk.specs),
        backend_name,
        device,
        options,
        subdir=SUBDIR,
    )


# ---------------------------------------------------------------------------
# the route's own counters
# ---------------------------------------------------------------------------
_LOCK = threading.RLock()
_COUNTERS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "load_failures": 0,
    "stores": 0,
    "save_failures": 0,
}


def _bump(name: str) -> None:
    with _LOCK:
        _COUNTERS[name] += 1


def reset_stats() -> None:
    """Zero this route's disk counters (the files are untouched)."""
    with _LOCK:
        for name in _COUNTERS:
            _COUNTERS[name] = 0


def stats() -> Dict[str, int]:
    """``batch_artifact_*`` counters of this route's disk tier.

    ``batch_artifact_enabled`` is 1 when ``EVOGP_EVOXC_ARTIFACT_DIR`` is set,
    ``batch_artifact_files`` the artifact files in the route's own region,
    ``batch_artifact_hits`` / ``batch_artifact_misses`` the bodies a build found on
    disk (a hit skips ``lower`` / ``compile`` / ``load``) versus did not,
    ``batch_artifact_load_failures`` the files present but unusable (discarded and
    rebuilt), and ``batch_artifact_stores`` / ``batch_artifact_save_failures`` the
    saves that succeeded / were skipped.  Deliberately separate from
    ``artifact_cache_stats()`` (the whole disk layer) and from
    ``batch_cache_stats()`` (the eight in-memory counters, pinned by the suite).
    """
    directory = artifact_cache.artifact_dir()
    region = os.path.join(directory, SUBDIR) if directory else None
    with _LOCK:
        counters = dict(_COUNTERS)
    snapshot = {
        "batch_artifact_enabled": int(directory is not None),
        "batch_artifact_files": artifact_cache.count_files(region),
    }
    snapshot.update(
        {f"batch_artifact_{name}": value for name, value in counters.items()}
    )
    return snapshot


# ---------------------------------------------------------------------------
# load / save (the engine's layer, this route's counters)
# ---------------------------------------------------------------------------
def load(
    path: Optional[str], specs: Sequence[Any], backend_name: str, device: Any
) -> Optional[Any]:
    """The artifact's executable, or ``None`` (never raises).

    ``None`` is also the "the tier is off" answer (``path is None``), counted as a
    miss so a record can distinguish "no file" from "not measured".  A file that is
    present but unusable is ``batch_artifact_load_failures`` -- the engine layer has
    already deleted it, so the rebuild can rewrite it.
    """
    if path is None:
        _bump("misses")
        return None
    try:
        present = os.path.isfile(path)
    except OSError:  # pragma: no cover - unreadable directory
        present = False
    executable = artifact_cache.load(path, specs, backend_name, device)
    if executable is not None:
        _bump("hits")
        return executable
    _bump("load_failures" if present else "misses")
    return None


def save(executable: Any, path: Optional[str]) -> bool:
    """Persist one compiled body; ``False`` when there was nothing to write.

    ``path is None`` (switch off) is a silent no-op that touches no counter --
    "disabled" and "the save failed" must never look alike.
    """
    if path is None:
        return False
    saved = artifact_cache.save(executable, path)
    _bump("stores" if saved else "save_failures")
    return saved


def load_body(
    plan: BatchPlan,
    chunk: BatchChunk,
    backend_name: str,
    device: Any,
    options: Any,
) -> Tuple[Optional[Any], Optional[str], float]:
    """Look one compiled body up on disk: ``(executable, path, load_ms)``.

    ``path`` is returned even on a miss, so the caller can hand it to
    :func:`save_body` after the build that the miss forced.  ``(None, None, 0.0)``
    when the tier is disabled.
    """
    path = body_path(plan, chunk, backend_name, device, options)
    if path is None:
        return None, None, 0.0
    started = time.perf_counter()
    executable = load(path, tuple(chunk.specs), backend_name, device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return executable, path, elapsed_ms


def save_body(executable: Any, path: Optional[str]) -> bool:
    """Persist a body that was just compiled (``<path>`` from :func:`load_body`)."""
    return save(executable, path)
