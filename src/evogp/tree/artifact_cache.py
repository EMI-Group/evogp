"""Opt-in on-disk executable artifacts for both evoxc evaluator routes.

:mod:`evogp.tree.evoxc_engine` and :mod:`evogp.tree.dense_engine` both build their
executables through ``evoxc.build_cached``, whose compiled results live in an
IN-MEMORY ``evoxc.CompileCache`` only -- nothing is shared across processes, so
every fresh process pays the compile again.  That is the cold-compile economics of
the engine: one dense compile is 3.1-3.4 s at ``L=32`` and 6.1 s at ``L=64``
(measured on this box), and the specialized route pays 271-284 ms PER distinct tree
structure.  This module adds the missing disk layer and nothing else: it turns an
``evoxc.Executable`` into a file and back, keyed by a stable string that means the
same thing in every process.

Switch
------
``EVOGP_EVOXC_ARTIFACT_DIR=/path`` enables the layer; unset/empty (the default)
disables it completely -- then :func:`artifact_path` returns ``None`` after one
environment read, so no key is derived, no directory is created and no file is
touched (exactly the engine's pre-artifact behaviour).  The directory is created on
the first successful save, never on a load miss.

Key
---
:func:`artifact_key` hashes everything that determines the compiled body, using
evoxc's own cache-key codec (:func:`evoxc.persist.cache.compute_key`, the same
``encode_value`` -> canonical-JSON -> sha256 walk ``build_cached`` uses) so that a
stable, process-independent string comes out:

* the graph's canonical name (``"dense:value"`` / ``"dense:loss"`` /
  ``"specialized"``) -- the placeholder for ``build_cached``'s ``id(fn)``, which
  differs in every process and is therefore unusable as a file name;
* the full ``*specs`` tuple (``TensorSpec`` shapes/dtypes AND the static content:
  the dense ``cfg`` / the specialized program tokens);
* the backend name, the device ``kind:index`` and the caller's frozen compile
  options (``evoxc_engine._freeze_options``);
* the versions of the two producers of the artifact: ``evoxc.__version__`` and the
  installed ``iree-base-compiler`` (empty when iree is not installed).

The result is ``"<readable-name>-<sha256 hex>"``.  Anything the codec cannot encode
(or any surprise while deriving it) makes :func:`artifact_path` return ``None``:
an underivable key disables the disk layer for that build instead of failing it.

Deliberately NOT in the key: the ambient ``EVOXC_*`` process-env compile knobs
(``EVOXC_TARGET_BACKENDS`` / ``EVOXC_OPT_LEVEL`` / ``EVOXC_IREE_*``).  They are not
part of the in-process build key either (evoxc documents them as knobs to be paired
with a cache clear); here the load path re-validates what actually matters -- the
iree adapter refuses an artifact whose compile targets do not match the requested
device kind, and :func:`signature_matches` refuses an artifact whose signature does
not fit the call shape -- and both refusals fall back to a real build.

Load / save contract
--------------------
:func:`load` returns a fully loaded ``evoxc.Executable`` or ``None``; it never
raises.  A missing file is a miss; an unreadable/corrupt/mismatched one is deleted
(best effort) and treated as a miss, so a stale artifact can only cost one rebuild,
never a wrong number.  :func:`save` writes to a temporary file in the same directory
and ``os.replace``-s it into place, so a concurrent or interrupted writer cannot
leave a torn artifact behind; it returns ``False`` (never raises) when the backend
executable has no ``save`` or the write fails -- an optional accelerator must never
break an evaluation.

Invalidation
------------
Two separate lifetimes, on purpose: :func:`evogp.tree.evoxc_engine.clear_cache`
(in-memory tables + ``evoxc.CompileCache``) does NOT touch the disk, and
:func:`clear` (``evoxc_engine.clear_artifacts()``) deletes the artifacts and resets
this module's counters without touching the in-memory tables.  :func:`stats`
(``evoxc_engine.artifact_cache_stats()``) is the disk layer's own counter dict --
it never reuses a :func:`evogp.tree.evoxc_engine.cache_stats` key.

What the layer is worth (measured on this box, ``Forest.SR_fitness``, dense+iree,
``EVOXC_DISABLE_LOCATIONS=1``): a ``p256``/``L=32``/``i8``/``N=1024`` compile is
3.34 s and its artifact 1.01 MiB, a second process loads it in 91 ms; a
``p64``/``L=64``/``i32``/``N=1024`` compile is 6.1 s and loads in ~0.1 s.  The load
is dominated by iree's own runtime/driver import, which any process pays anyway.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import evoxc
from evoxc.persist.cache import compute_key

__all__ = [
    "DIR_ENV",
    "SUFFIX",
    "artifact_dir",
    "artifact_key",
    "artifact_path",
    "clear",
    "load",
    "save",
    "signature_matches",
    "reset_stats",
    "stats",
]

#: Enable switch: the artifact directory.  Unset/empty = the disk layer is off.
DIR_ENV = "EVOGP_EVOXC_ARTIFACT_DIR"
#: Suffix of every artifact file (``clear`` only ever deletes these).
SUFFIX = ".evoxcart"

#: First key component: bump it when the key derivation itself changes (a new
#: component, a different name spelling) so old files can never be mistaken for
#: new ones.  The evoxc codec already versions its own encoding
#: (``EVOXC_FORMAT_VERSION`` participates inside ``compute_key``).
KEY_FORMAT = "evogp-executable-artifact-v1"


def artifact_dir() -> Optional[str]:
    """The enabled artifact directory, or ``None`` (the disk layer is off)."""
    raw = os.environ.get(DIR_ENV, "").strip()
    return raw or None


# ---------------------------------------------------------------------------
# key derivation
# ---------------------------------------------------------------------------


def _evoxc_version() -> str:
    """``evoxc.__version__`` (``""`` when the attribute is missing)."""
    return str(getattr(evoxc, "__version__", "") or "")


def _iree_version() -> str:
    """Installed ``iree-base-compiler`` / ``iree-compiler`` version, else ``""``.

    Read through :mod:`importlib.metadata` so nothing imports iree: the version is
    the compile-stage producer of the artifact, and a compiler upgrade must
    invalidate old files.  Only the *first* importable distribution name wins.
    """
    try:
        from importlib.metadata import version

        for name in ("iree-base-compiler", "iree-compiler"):
            try:
                return str(version(name))
            except Exception:  # not installed under this name
                continue
    except Exception:  # pragma: no cover - importlib.metadata always present
        pass
    return ""


def _device_tag(device: Any) -> str:
    """``"kind:index"`` for an ``evoxc.core.Device`` (``str(device)`` otherwise).

    The kind+index pair is what selects the HAL driver and the physical device, and
    it is exactly what the codec encodes for a ``Device``; spelling it out keeps the
    key readable and independent of evoxc's ``Device.__repr__``.
    """
    kind = getattr(device, "kind", None)
    if kind is None:
        return str(device)
    return f"{kind}:{getattr(device, 'index', 0)}"


def _slug(name: str) -> str:
    """File-name-safe readable prefix of a graph name (``"dense:loss"``)."""
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "-" for char in str(name)
    )
    return cleaned.strip("-") or "graph"


def artifact_key(
    name: str,
    specs: Sequence[Any],
    backend_name: str,
    device: Any,
    options: Any,
) -> str:
    """Stable ``"<readable-name>-<sha256 hex>"`` key of one executable.

    ``specs`` is the exact spec tuple the route passes to ``evoxc.build_cached``
    (tensor specs AND static content), ``options`` its frozen compile options.
    Raises whatever ``compute_key`` raises on an unencodable component -- callers
    use :func:`artifact_path`, which turns that into "no artifact".
    """
    digest = compute_key(
        [
            KEY_FORMAT,
            str(name),
            tuple(specs),
            str(backend_name),
            _device_tag(device),
            options,
            _evoxc_version(),
            _iree_version(),
        ]
    )
    return f"{_slug(name)}-{digest}"


def artifact_path(
    name: str,
    specs: Sequence[Any],
    backend_name: str,
    device: Any,
    options: Any,
) -> Optional[str]:
    """Artifact file path of this build, or ``None`` (disabled / underivable).

    ``None`` is the "do nothing" answer: the switch is off, or the key could not be
    derived (an unhashable/uncodable spec), in which case the caller just builds.
    """
    directory = artifact_dir()
    if directory is None:
        return None
    try:
        key = artifact_key(name, specs, backend_name, device, options)
    except Exception:
        return None
    return os.path.join(directory, key + SUFFIX)


# ---------------------------------------------------------------------------
# signature validation (a loaded file must fit the call shape)
# ---------------------------------------------------------------------------


def _flatten_statics(value: Any, out: List[Any]) -> None:
    """Append the leaves of ``value`` to ``out`` (tuple/list descend, else leaf).

    Mirrors the pre-order leaf walk evoxc's trace records in
    ``Signature.static_values`` for the static parts of a spec tuple: a nested
    tuple/list (the dense ``cfg``'s ``used_func_ids``) contributes its elements,
    every other value is one leaf.
    """
    if isinstance(value, (tuple, list)):
        for item in value:
            _flatten_statics(item, out)
    else:
        out.append(value)


def _same(left: Any, right: Any) -> bool:
    """Scalar equality that answers ``False`` instead of raising/arrays."""
    try:
        equal = left == right
    except Exception:  # pragma: no cover - exotic comparison failures
        return False
    return bool(equal) if isinstance(equal, bool) else False


def signature_matches(executable: Any, specs: Sequence[Any]) -> bool:
    """True when ``executable``'s recorded signature fits ``specs``.

    The check is the cheap half of a rebuild: every tensor leaf of ``specs``
    (shape + dtype, in order) must equal the executable's ``signature.input_specs``,
    and the static leaves must equal its ``signature.static_values`` (flattened in
    pre-order, see :func:`_flatten_statics`).  Anything unexpected -- a missing
    signature, an unreadable spec, a comparison that does not answer a ``bool`` --
    returns ``False``, which the caller treats as a corrupt artifact: delete and
    rebuild.  The static comparison is the belt to the key's braces; the key already
    covers both halves.
    """
    signature = getattr(executable, "signature", None)
    if signature is None:
        return False
    try:
        expected_tensors: List[Tuple[Tuple[int, ...], str]] = []
        expected_statics: List[Any] = []
        for spec in specs:
            if isinstance(spec, evoxc.TensorSpec):
                expected_tensors.append((tuple(int(d) for d in spec.shape), str(spec.dtype)))
            else:
                _flatten_statics(spec, expected_statics)
        actual_tensors = [
            (tuple(int(d) for d in spec.shape), str(spec.dtype))
            for spec in getattr(signature, "input_specs", ())
        ]
        actual_statics = list(getattr(signature, "static_values", ()))
    except Exception:  # pragma: no cover - malformed signature object
        return False
    if expected_tensors != actual_tensors:
        return False
    if len(expected_statics) != len(actual_statics):
        return False
    return all(_same(a, b) for a, b in zip(expected_statics, actual_statics))


# ---------------------------------------------------------------------------
# counters
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
    """Zero the disk layer's counters (the files are untouched)."""
    with _LOCK:
        for name in _COUNTERS:
            _COUNTERS[name] = 0


def _count_files(directory: Optional[str]) -> int:
    """Artifact files currently in ``directory`` (0 when it does not exist)."""
    if directory is None:
        return 0
    try:
        return sum(1 for entry in os.listdir(directory) if entry.endswith(SUFFIX))
    except OSError:
        return 0


def stats() -> Dict[str, int]:
    """Snapshot of the disk layer: ``artifact_*`` counters and file count.

    ``artifact_enabled`` is 1 when ``EVOGP_EVOXC_ARTIFACT_DIR`` is set, else 0;
    ``artifact_files`` is the number of artifact files in that directory;
    ``artifact_hits`` / ``artifact_misses`` count :func:`load` calls (a hit returned
    a usable executable, a miss found no file), ``artifact_load_failures`` counts
    files that were present but unusable (deleted + rebuilt), ``artifact_stores``
    successful :func:`save` calls and ``artifact_save_failures`` the ones that were
    skipped or failed.  The keys are disjoint from
    :func:`evogp.tree.evoxc_engine.cache_stats` and
    :func:`evogp.tree.dense_engine.dense_cache_stats`.
    """
    directory = artifact_dir()
    with _LOCK:
        counters = dict(_COUNTERS)
    snapshot = {
        "artifact_enabled": int(directory is not None),
        "artifact_files": _count_files(directory),
    }
    snapshot.update({f"artifact_{name}": value for name, value in counters.items()})
    return snapshot


# ---------------------------------------------------------------------------
# load / save / clear
# ---------------------------------------------------------------------------


def _discard(path: str) -> None:
    """Delete an unusable artifact (best effort; never raises)."""
    try:
        os.unlink(path)
    except OSError:
        pass


def load(
    path: str,
    specs: Sequence[Any],
    backend_name: str,
    device: Any,
) -> Optional[Any]:
    """The artifact's executable, or ``None`` (never raises).

    ``None`` means "build it": the file is absent (a miss) or present but unusable
    -- corrupt, from another backend/device/version, or its signature does not fit
    ``specs`` -- in which case it is deleted first, so the following :func:`save` can
    rewrite it.  Every exception evoxc's loader can raise (``PersistenceError`` for a
    format/version/backend mismatch, ``BackendError`` for a device/target mismatch,
    ``OSError`` for a truncated file) lands in the same place: rebuild.
    """
    try:
        present = os.path.isfile(path)
    except OSError:  # pragma: no cover - unreadable directory
        present = False
    if not present:
        _bump("misses")
        return None
    try:
        executable = evoxc.Executable.load(path, backend=backend_name, device=device)
        usable = signature_matches(executable, specs)
    except Exception:
        _bump("load_failures")
        _discard(path)
        return None
    if not usable:
        _bump("load_failures")
        _discard(path)
        return None
    _bump("hits")
    return executable


def save(executable: Any, path: str) -> bool:
    """Persist ``executable`` to ``path`` atomically; ``False`` when not saved.

    Written to ``<path>.tmp-<pid>-<uuid>`` in the same directory and renamed into
    place, so a concurrent reader either sees the previous complete artifact or the
    new one -- never a partially written file.  A backend executable without a
    usable ``save`` (evoxc's shared ``CompilerExecutable.save`` needs a recorded
    ``CompiledArtifact``; some backends have none) raises inside the try and is
    reported as ``False``: saving is best effort and never breaks a call.
    """
    directory = os.path.dirname(path) or "."
    temporary = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        os.makedirs(directory, exist_ok=True)
        executable.save(temporary)
        os.replace(temporary, path)
    except Exception:
        _bump("save_failures")
        _discard(temporary)
        return False
    _bump("stores")
    return True


def clear() -> None:
    """Delete every artifact file in the enabled directory and reset counters.

    The disk layer's own invalidation (``evoxc_engine.clear_artifacts()``): it is
    deliberately NOT part of :func:`evogp.tree.evoxc_engine.clear_cache`, whose
    meaning stays "the in-memory tables".  A no-op when the switch is off; only
    ``*.evoxcart`` files are deleted, so a directory shared with other data is safe.
    """
    directory = artifact_dir()
    if directory is not None:
        try:
            entries = os.listdir(directory)
        except OSError:
            entries = []
        for entry in entries:
            if entry.endswith(SUFFIX) or f"{SUFFIX}.tmp-" in entry:
                _discard(os.path.join(directory, entry))
    reset_stats()

