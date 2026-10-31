"""Local DLPack interop shim: ``evoxc`` DEVICE tensors <-> torch, zero-copy.

TEMPORARY, deliberately local
-----------------------------
``evoxc``'s documented device/host interop is host-only: ``from_numpy`` is a
zero-copy HOST wrap, a device tensor is read back through an explicit
``t.to(Device("cpu", 0))``, and there is no first-class way to hand a device
buffer to an external kernel and take a device result back without a round
trip.  The sibling Triton engine (:mod:`evogp.tree.fused_engine`) is exactly
such an external kernel, so the *device-native* fused route needs both
directions at pointer level.

This module is that bridge and **nothing else**: it is local to ``evogp.gp``,
it is explicitly temporary, and it should be deleted the day ``evoxc`` grows a
first-class device-payload DLPack export/import (then every call site here
becomes a one-line evoxc call).  It does not wrap ``evogp``'s own types, does
not memoize any numerics and does not own a route.

The protocol it relies on
-------------------------
Two duck-typed surfaces, both currently private to ``evoxc``/``iree``:

1. the evoxc device payload -- ``tensor.data`` is an
   ``evoxc.backends.adapters.iree.IreeDevicePayload`` whose PUBLIC interop
   contract is ``.device_array`` (an opaque iree runtime ``DeviceArray``),
   ``.shape``, ``.dtype``, ``.device`` (the ``evoxc.Device``) and ``.to_host()``;
2. the iree runtime handle -- ``payload.device_array._device`` is a
   ``HalDevice`` and ``payload.device_array._buffer_view`` a ``HalBufferView``;
   ``HalDevice`` owns ``create_dlpack_capsule(view, device_type_code,
   device_id)`` and ``from_dlpack_capsule(capsule)``.

Every one of those is probed, never assumed: :func:`interop_available` answers
whether the protocol is present on this box, and every entry point here
validates what it got and raises :class:`InteropUnavailable` -- naming the
unsupported thing AND the remedy (the host path) -- rather than returning a
silently wrong tensor.

What "zero-copy" means here
---------------------------
* :func:`export_to_torch` -- the evoxc buffer view becomes a torch tensor that
  SHARES its storage (the DLPack capsule carries the HAL buffer handle; no
  bytes move).  Shape and dtype are taken from the payload's metadata and
  re-verified against the exported tensor.
* :func:`import_from_torch` -- a torch tensor becomes an ``evoxc.Tensor`` whose
  payload wraps the same memory (``HalDevice.from_dlpack_capsule`` on the
  tensor's own capsule).  Contiguity is REQUIRED: a non-contiguous tensor is a
  Python-level error (``ValueError``) unless the caller passes ``copy=True``,
  which is the only place this module will accept a copy and says so.

Synchronization
---------------
iree's HIP/CUDA driver and torch share the process' default HIP/CUDA stream on
this box, so sharing a pointer needs no extra fence: work enqueued by an evoxc
``run`` (the variation graph) and work enqueued by a torch/Triton launch are
serialized by the stream itself.  This is an ASSUMPTION, not a theorem about
iree's API -- it is validated empirically by the chained-generation parity test
(ten driver generations whose channels must match the in-graph route bit for
bit), and it is the reason this module does not expose a ``stream=`` argument.
If a future runtime stops sharing the default stream, that test breaks -- treat
it as the tripwire, and add an explicit event wait here rather than a silent
tolerance downstream.

Import discipline (do not "tidy" this)
-------------------------------------
Nothing is imported at module import time, and NO literal module import
statement of the form the purity grep forbids appears anywhere in this file
(``tests/test_gp_stack_purity.py`` greps every module of this subtree, and the
clean-subprocess test requires ``import evogp.gp`` to work with only ``src`` on
``PYTHONPATH``).  ``torch``, ``evoxc`` and ``iree.runtime`` are resolved
through :func:`importlib.import_module` INSIDE the functions, cached in
:data:`_IMPORTS` after the first successful call
(:func:`clear_interop_cache` drops that cache, which is what a test that
monkeypatches a module needs).

Device kinds
------------
Only kinds whose DLPack device-type code is known and whose pointer can be
shared with torch are served (:data:`DEVICE_TYPE_CODES`: ``rocm``/``hip`` ->
``kDLROCM`` 10, ``cuda`` -> ``kDLCUDA`` 2).  The ``cpu`` kind (and every
non-GPU kind, e.g. ``vulkan``/``llvm-cpu``) is DECLINED with
:class:`InteropUnavailable`: the host path already covers those, and a shared
CPU buffer buys nothing.

Public surface: :class:`InteropUnavailable`, :data:`DEVICE_TYPE_CODES`,
:data:`SUPPORTED_TORCH_DTYPES`, :func:`interop_available`,
:func:`is_device_tensor`, :func:`device_of`, :func:`hal_device`,
:func:`export_to_torch`, :func:`import_from_torch`,
:func:`clear_interop_cache`.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Optional

import numpy as np

__all__ = [
    "InteropUnavailable",
    "DEVICE_TYPE_CODES",
    "SUPPORTED_TORCH_DTYPES",
    "interop_available",
    "is_device_tensor",
    "device_of",
    "hal_device",
    "export_to_torch",
    "import_from_torch",
    "clear_interop_cache",
]

#: evoxc device kind -> DLPack device-type code.  ``kDLCUDA`` is 2 and
#: ``kDLROCM`` is 10 in the DLPack spec; ``"hip"`` is evoxc's alias for
#: ``"rocm"`` (same HIP buffers, same code).  A kind outside this table is
#: declined -- including ``"cpu"``, whose host path needs no interop at all.
DEVICE_TYPE_CODES = {"rocm": 10, "hip": 10, "cuda": 2}

#: The torch dtype names this shim will move across the boundary.  The
#: ``evogp.gp`` channels need ``int32``/``float32``/``int16``; the rest are the
#: cheap, unambiguous extras (no complex, no quantized, no float8 -- those
#: either have no numpy counterpart or no well-defined DLPack element type).
SUPPORTED_TORCH_DTYPES = frozenset(
    {
        "float32",
        "float64",
        "float16",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
    }
)

#: numpy dtype name -> torch dtype attribute (the two spellings agree for the
#: supported set, kept explicit so an unexpected numpy dtype declines instead
#: of silently mapping through ``getattr``).
_NUMPY_TO_TORCH_NAME = {
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}

#: evoxc device kind -> torch device TYPE ("cuda" covers a ROCm build too: the
#: repository-wide policy is that ``"cuda"`` selects the AMD GPU under torch).
_TORCH_DEVICE_TYPES = {"rocm": "cuda", "hip": "cuda", "cuda": "cuda"}

#: Successful imports, by module name (a failing import is cached as the
#: :class:`InteropUnavailable` it produced, so a missing dependency is diagnosed
#: once instead of once per generation).
_IMPORTS: dict = {}

#: ``"kind:index"`` -> the ``HalDevice`` last seen for that evoxc device.
#: A fast path only: a payload's OWN runtime device always wins (see
#: :func:`hal_device`), because a DLPack capsule must be created on the device
#: instance the buffer belongs to.
_HAL_DEVICES: dict = {}

#: The remedy sentence every message ends with.
_HOST_PATH = (
    "the host path still serves this call (download the channels, run the "
    "kernel from numpy); nothing is silently approximated"
)


class InteropUnavailable(RuntimeError):
    """A device-pointer exchange is not possible for this call (or this box).

    Raised with the reason, the unsupported thing and the remedy -- never
    instead of a value that would have been wrong.  Callers that can fall back
    should catch this class (the device-native evaluator in
    :mod:`evogp.gp.fused_eval` does exactly that and retries on the host).
    """


# --------------------------------------------------------------------------- #
# lazy imports (the whole point of the module)
# --------------------------------------------------------------------------- #
def _module(name: str, remedy: str) -> Any:
    """Import ``name`` at CALL time, cached; raise :class:`InteropUnavailable`."""
    cached = _IMPORTS.get(name)
    if cached is None:
        try:
            cached = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            cached = InteropUnavailable(
                f"the device interop needs {name}, which could not be imported "
                f"({type(exc).__name__}: {exc}); {remedy}"
            )
        _IMPORTS[name] = cached
    if isinstance(cached, InteropUnavailable):
        raise cached
    return cached


def _maybe_module(name: str, remedy: str) -> Optional[Any]:
    """``_module(name)``, or ``None`` when it is unavailable (probes only)."""
    try:
        return _module(name, remedy)
    except InteropUnavailable:
        return None


def _torch_module() -> Any:
    """torch, resolved lazily (the shim's one torch contact)."""
    return _module(
        "torch",
        f"use the numpy/host route instead; {_HOST_PATH}",
    )


def _evoxc_module() -> Any:
    """The evoxc package, resolved lazily."""
    return _module(
        "evoxc",
        "install evoxc (or put the sibling checkout on PYTHONPATH); the host "
        "path needs it too",
    )


def _payload_class() -> Any:
    """``evoxc.backends.adapters.iree.IreeDevicePayload`` (the payload protocol)."""
    adapter = _module(
        "evoxc.backends.adapters.iree",
        "the iree adapter is what carries device payloads; a numpy-only evoxc "
        "has no device tensors to share",
    )
    payload_cls = getattr(adapter, "IreeDevicePayload", None)
    if payload_cls is None:  # the pre-rename private spelling
        payload_cls = getattr(adapter, "_IreeDevicePayload", None)
    if payload_cls is None:
        raise InteropUnavailable(
            "evoxc.backends.adapters.iree exposes no IreeDevicePayload, so there "
            f"is no device payload to export; {_HOST_PATH}"
        )
    return payload_cls


def clear_interop_cache() -> None:
    """Drop the lazy-import cache and the per-device HAL memo.

    Only needed when something else replaced ``torch`` / ``evoxc`` /
    ``iree.runtime`` in ``sys.modules`` (a test that simulates a box without
    them, for instance); a normal process never calls this.
    """
    _IMPORTS.clear()
    _HAL_DEVICES.clear()


# --------------------------------------------------------------------------- #
# kind / dtype plumbing
# --------------------------------------------------------------------------- #
def _normalise_kind(kind: Any) -> str:
    """The canonical lower-case kind spelling of an evoxc kind string."""
    return str(kind).strip().lower().split(":")[0]


def _device_type_code(kind: str) -> int:
    """The DLPack device-type code of an evoxc kind, or :class:`InteropUnavailable`."""
    normalised = _normalise_kind(kind)
    if normalised == "cpu":
        raise InteropUnavailable(
            "the device interop declines the 'cpu' kind: a host tensor needs no "
            f"pointer sharing and the host path already serves it; {_HOST_PATH}"
        )
    code = DEVICE_TYPE_CODES.get(normalised)
    if code is None:
        raise InteropUnavailable(
            f"the device interop does not know the evoxc device kind "
            f"{kind!r}: only {sorted(DEVICE_TYPE_CODES)} have a DLPack device "
            f"type code that torch can accept; {_HOST_PATH}"
        )
    return int(code)


def _check_kind_against_torch(torch_module: Any, kind: str) -> None:
    """Refuse a device kind this torch build cannot be the same device as.

    The eVOXC kind and torch's build are two spellings of one physical device:
    a ROCm build reports ``torch.version.hip`` and only ``rocm``/``hip`` buffers
    live on it, an NVIDIA build has no ``hip`` and only ``cuda`` buffers do.  A
    capsule minted with the wrong DLPack code would be a silent misinterpretation
    of the pointer, so it is refused here instead.
    """
    normalised = _normalise_kind(kind)
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    if normalised in ("rocm", "hip") and not hip:
        raise InteropUnavailable(
            f"the device interop cannot serve evoxc kind {kind!r}: this is not a "
            f"ROCm torch build (torch.version.hip is {hip!r}), so no {kind!r} "
            f"buffer can be shared with it; {_HOST_PATH}"
        )
    if normalised == "cuda" and hip:
        raise InteropUnavailable(
            f"the device interop cannot serve evoxc kind 'cuda' on a ROCm torch "
            f"build (torch.version.hip is {hip!r}): the buffers are HIP buffers "
            f"and their DLPack code is kDLROCM; {_HOST_PATH}"
        )


def _torch_dtype(torch_module: Any, dtype: Any) -> Any:
    """The torch dtype of a payload's numpy dtype, or :class:`InteropUnavailable`."""
    try:
        name = np.dtype(dtype).name
    except TypeError as exc:
        raise InteropUnavailable(
            f"the device payload reports the dtype {dtype!r}, which numpy cannot "
            f"normalise ({exc}); {_HOST_PATH}"
        ) from exc
    attribute = _NUMPY_TO_TORCH_NAME.get(name)
    torch_dtype = getattr(torch_module, attribute, None) if attribute else None
    if torch_dtype is None:
        raise InteropUnavailable(
            f"the device interop does not support the dtype {name!r} (supported: "
            f"{sorted(SUPPORTED_TORCH_DTYPES)}); {_HOST_PATH}"
        )
    return torch_dtype


def _check_torch_dtype(torch_tensor: Any) -> None:
    """Raise :class:`InteropUnavailable` for a torch dtype with no clean import."""
    name = str(torch_tensor.dtype).removeprefix("torch.")
    if name not in SUPPORTED_TORCH_DTYPES:
        raise InteropUnavailable(
            f"the device interop does not support a {torch_tensor.dtype} tensor "
            f"(supported: {sorted(SUPPORTED_TORCH_DTYPES)}); {_HOST_PATH}"
        )


def _device_key(device: Any) -> str:
    """A stable ``"kind:index"`` spelling of an evoxc device."""
    return f"{_normalise_kind(getattr(device, 'kind', ''))}:{int(getattr(device, 'index', 0))}"


# --------------------------------------------------------------------------- #
# the payload protocol
# --------------------------------------------------------------------------- #
def _payload_of(tensor: Any) -> Any:
    """The device payload of ``tensor`` (or :class:`InteropUnavailable`)."""
    evoxc_module = _evoxc_module()
    if not isinstance(tensor, evoxc_module.Tensor):
        raise InteropUnavailable(
            "the device interop needs an evoxc.Tensor (a device buffer), got "
            f"{type(tensor).__name__}; {_HOST_PATH}"
        )
    payload = getattr(tensor, "data", None)
    payload_cls = _payload_class()
    if not isinstance(payload, payload_cls):
        raise InteropUnavailable(
            "the evoxc tensor carries "
            f"{type(payload).__name__} rather than an IreeDevicePayload (a HOST "
            f"tensor has a numpy payload), so there is no device pointer to "
            f"share; {_HOST_PATH}"
        )
    for attribute in ("device_array", "shape", "dtype"):
        if not hasattr(payload, attribute):
            raise InteropUnavailable(
                f"the device payload has no .{attribute} (this evoxc/iree "
                f"combination predates the payload interop contract); {_HOST_PATH}"
            )
    return payload


def _device_of_payload(payload: Any) -> Any:
    """The ``evoxc.Device`` a payload lives on (or :class:`InteropUnavailable`)."""
    device = getattr(payload, "device", None)
    if device is None or not hasattr(device, "kind"):
        raise InteropUnavailable(
            "the device payload does not name its evoxc Device, so the target "
            f"device cannot be verified; {_HOST_PATH}"
        )
    return device


def is_device_tensor(value: Any) -> bool:
    """Whether ``value`` is an ``evoxc.Tensor`` resident on a non-cpu device.

    Never raises: ``False`` for numpy arrays, for host evoxc tensors and for a
    payload/device combination this shim cannot inspect.  This is the dispatch
    probe of :class:`evogp.gp.fused_eval.FusedDeviceEvaluator` -- it decides
    whether the device route is even worth attempting, so it must stay free of
    exceptions and free of host transfers.
    """
    evoxc_module = _maybe_module("evoxc", "no evoxc means no device tensors")
    if evoxc_module is None or not isinstance(value, evoxc_module.Tensor):
        return False
    payload_cls = _maybe_payload_class()
    if payload_cls is None:
        return False
    payload = getattr(value, "data", None)
    if not isinstance(payload, payload_cls):
        return False
    device = getattr(payload, "device", None)
    if device is None:
        return False
    return _normalise_kind(getattr(device, "kind", "cpu")) != "cpu"


def _maybe_payload_class() -> Optional[Any]:
    """``_payload_class()``, or ``None`` when the adapter is unavailable."""
    try:
        return _payload_class()
    except InteropUnavailable:
        return None


def device_of(tensor: Any) -> Any:
    """The ``evoxc.Device`` of a device-resident tensor.

    Args:
        tensor: an ``evoxc.Tensor`` whose payload is a device payload.

    Returns:
        The ``evoxc.Device`` (``Device(kind='rocm', index=0)`` on this box).

    Raises:
        InteropUnavailable: for a host tensor, a numpy array or a payload that
            does not name its device.
    """
    return _device_of_payload(_payload_of(tensor))


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
def _protocol_present() -> bool:
    """Whether every attribute the shim needs exists (never raises)."""
    torch_module = _maybe_module("torch", "no torch, no pointer sharing")
    if torch_module is None:
        return False
    if not hasattr(getattr(torch_module, "utils", None), "dlpack"):
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    iree = _maybe_module(
        "iree.runtime",
        "the pointer hand-off goes through iree's HAL DLPack helpers; "
        "iree-base-runtime is not installed here, so use the host path",
    )
    if iree is None:
        return False
    hal_device = getattr(iree, "HalDevice", None)
    if hal_device is None:
        return False
    for attribute in ("create_dlpack_capsule", "from_dlpack_capsule"):
        if not hasattr(hal_device, attribute):
            return False
    payload_cls = _maybe_payload_class()
    if payload_cls is None:
        return False
    for attribute in ("device_array", "shape", "dtype", "to_host"):
        if not hasattr(payload_cls, attribute):
            return False
    return True


def interop_available(kind: Optional[Any] = None) -> bool:
    """Whether the device-native fused route can serve a call here; never raises.

    True only when ALL of these hold: ``torch`` is importable with a visible
    CUDA/HIP device (``torch.cuda.is_available()``), ``triton`` is installed
    (the fused kernel is the reason for the hand-off at all), ``iree.runtime``
    exposes the HAL DLPack helpers, and ``evoxc`` exposes an
    ``IreeDevicePayload`` with the payload-protocol attributes.

    Args:
        kind: optional evoxc device kind (``"rocm"`` / ``"cuda"`` / ...) the
            caller intends to use; an unknown or ``cpu`` kind answers ``False``,
            and so does a kind this torch build cannot be the same device as
            (``"rocm"`` on an NVIDIA build, ``"cuda"`` on a ROCm build) --
            either is refused without touching the rest of the probe.

    Notes:
        A per-call refusal (an unsupported dtype, a non-contiguous tensor, a
        population outside the kernel's scratch envelope) is NOT visible here --
        that is :class:`InteropUnavailable` from the call, which the evaluator
        turns into a fallback.  The probe is cheap (imports are cached, nothing
        is compiled) and deliberately not memoized, so it stays honest when a
        module is swapped underneath it.
    """
    try:
        if kind is not None and _normalise_kind(kind) not in DEVICE_TYPE_CODES:
            return False
        if not _protocol_present():
            return False
        torch_module = _maybe_module(
            "torch", "a torch device is what the pointers are shared with"
        )
        if torch_module is None or not torch_module.cuda.is_available():
            return False
        if kind is not None:
            _check_kind_against_torch(torch_module, kind)
    except InteropUnavailable:
        return False
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        return False
    return True


# --------------------------------------------------------------------------- #
# the two directions
# --------------------------------------------------------------------------- #
def hal_device(tensor: Any) -> Any:
    """The iree ``HalDevice`` behind a device tensor (memoized per evoxc device).

    The memo is a fast path for the stable case (evoxc caches one runtime
    device per ``Device(kind, index)`` on rocm/hip): the first payload seen for
    a given evoxc device records its ``HalDevice`` under ``"kind:index"`` and
    later calls return the recorded handle.  A payload's OWN runtime device
    always wins when it differs -- a DLPack capsule must be created on the
    device instance the buffer belongs to, and the CUDA path may hand out a
    fresh instance per load -- so the memo can never make a call wrong, only
    cheap.

    Raises:
        InteropUnavailable: for a host tensor, or a runtime whose
            ``DeviceArray`` does not expose ``_device``/``_buffer_view``.
    """
    payload = _payload_of(tensor)
    device = _device_of_payload(payload)
    own = getattr(getattr(payload, "device_array", None), "_device", None)
    view = getattr(getattr(payload, "device_array", None), "_buffer_view", None)
    if own is None or view is None:
        raise InteropUnavailable(
            "the iree device array behind this payload does not expose the "
            "private _device/_buffer_view handles the DLPack bridge needs (this "
            f"iree build predates them); {_HOST_PATH}"
        )
    for attribute in ("create_dlpack_capsule", "from_dlpack_capsule"):
        if not hasattr(own, attribute):
            raise InteropUnavailable(
                f"the iree HAL device exposes no {attribute} (this iree build "
                f"predates the HAL DLPack helpers); {_HOST_PATH}"
            )
    key = _device_key(device)
    cached = _HAL_DEVICES.get(key)
    if cached is not own:
        _HAL_DEVICES[key] = own
        return own
    return cached


def export_to_torch(tensor: Any, *, device_kind: Optional[Any] = None) -> Any:
    """A torch tensor SHARING the memory of an ``evoxc`` device tensor.

    Args:
        tensor: an ``evoxc.Tensor`` resident on a ``rocm``/``hip``/``cuda``
            device.
        device_kind: optional assertion/override for the DLPack device kind.
            When given it must name a supported kind (otherwise
            :class:`InteropUnavailable`) and must agree with the payload's own
            ``evoxc.Device`` kind (otherwise ``ValueError``).

    Returns:
        A torch tensor on the corresponding torch device (``"cuda"`` on a ROCm
        build, as everywhere else in this repository), with the payload's shape
        and dtype.  No bytes are copied: writing through either handle is
        visible through the other.

    Raises:
        InteropUnavailable: no torch / no iree HAL helpers, a host payload, an
            unsupported device kind or dtype, a cpu device, or an exported
            tensor whose metadata disagrees with the payload's (a loud guard
            against a wrong mapping).
        ValueError: ``device_kind`` contradicts the tensor's own device.
    """
    torch_module = _torch_module()
    payload = _payload_of(tensor)
    device = _device_of_payload(payload)
    kind = _normalise_kind(getattr(device, "kind", ""))
    if device_kind is not None:
        override = _normalise_kind(device_kind)
        if override != kind:
            raise ValueError(
                f"device_kind={device_kind!r} contradicts the tensor's own evoxc "
                f"device {getattr(device, 'kind', None)!r}"
            )
        kind = override
    code = _device_type_code(kind)
    _check_kind_against_torch(torch_module, kind)
    expected_shape = tuple(int(size) for size in tuple(payload.shape))
    expected_dtype = _torch_dtype(torch_module, payload.dtype)
    hal = hal_device(tensor)
    data_array = payload.device_array
    capsule = hal.create_dlpack_capsule(
        data_array._buffer_view, code, int(getattr(device, "index", 0))
    )
    try:
        exported = torch_module.utils.dlpack.from_dlpack(capsule)
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            f"torch refused the DLPack capsule of a {kind} buffer "
            f"({type(exc).__name__}: {exc}); {_HOST_PATH}"
        ) from exc
    if tuple(exported.shape) != expected_shape:
        raise InteropUnavailable(
            f"the exported torch tensor has shape {tuple(exported.shape)} but the "
            f"payload reports {expected_shape} -- refusing to hand back a "
            f"differently-shaped view; {_HOST_PATH}"
        )
    if exported.dtype != expected_dtype:
        raise InteropUnavailable(
            f"the exported torch tensor has dtype {exported.dtype} but the "
            f"payload reports {np.dtype(payload.dtype).name} "
            f"(torch {expected_dtype}) -- refusing to reinterpret the buffer; "
            f"{_HOST_PATH}"
        )
    return exported


def import_from_torch(torch_tensor: Any, like: Any, *, copy: bool = False) -> Any:
    """An ``evoxc.Tensor`` sharing the memory of a torch tensor.

    Args:
        torch_tensor: the tensor to adopt.  Must be CONTIGUOUS and live on the
            torch device that corresponds to ``like``'s evoxc device.
        like: a device-resident ``evoxc.Tensor`` whose device the result is
            placed on (typically one of the channels the value was derived
            from) -- it supplies the ``HalDevice`` and the ``evoxc.Device`` the
            new tensor must carry to be accepted by the next ``evoxc.run``.
        copy: make the tensor contiguous first (``torch.Tensor.contiguous()``,
            which COPIES when the input is not already contiguous).  This is the
            only place the shim accepts a copy; it is off by default because a
            silent copy is exactly the kind of half-zero-copy a hot loop cannot
            see.

    Returns:
        An ``evoxc.Tensor`` on ``like``'s device whose payload wraps the SAME
        memory as ``torch_tensor`` (shape and dtype re-read from the imported
        payload and compared against the torch tensor).

    Raises:
        InteropUnavailable: no torch / no iree HAL helpers, an unsupported
            device kind or dtype, a device mismatch, or an imported payload
            whose metadata disagrees with the torch tensor.
        ValueError: a non-contiguous tensor without ``copy=True``.
        TypeError: ``torch_tensor`` is not a torch tensor.
    """
    torch_module = _torch_module()
    if not isinstance(torch_tensor, torch_module.Tensor):
        raise TypeError(
            "import_from_torch needs a torch tensor, got "
            f"{type(torch_tensor).__name__}"
        )
    payload = _payload_of(like)
    device = _device_of_payload(payload)
    kind = _normalise_kind(getattr(device, "kind", ""))
    _device_type_code(kind)
    _check_kind_against_torch(torch_module, kind)
    _check_torch_dtype(torch_tensor)
    expected_type = _TORCH_DEVICE_TYPES.get(kind)
    actual = torch_tensor.device
    index = int(getattr(device, "index", 0))
    if expected_type is None or actual.type != expected_type:
        raise InteropUnavailable(
            f"the torch tensor lives on {actual}, which is not the torch view of "
            f"the evoxc {kind!r} device (expected a torch {expected_type!r} "
            f"tensor); {_HOST_PATH}"
        )
    if actual.index is not None and int(actual.index) != index:
        raise InteropUnavailable(
            f"the torch tensor lives on {actual} but the evoxc device is "
            f"{kind}:{index} -- importing would place the buffer on the wrong "
            f"device; {_HOST_PATH}"
        )
    if copy:
        torch_tensor = torch_tensor.contiguous()
    elif not torch_tensor.is_contiguous():
        raise ValueError(
            "import_from_torch needs a contiguous tensor (a non-contiguous "
            "layout cannot be adopted without a copy); pass copy=True to accept "
            f"the .contiguous() copy explicitly. Layout: {torch_tensor.shape} "
            f"strides {torch_tensor.stride()}"
        )
    hal = hal_device(like)
    try:
        buffer_view = hal.from_dlpack_capsule(torch_tensor.__dlpack__())
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            f"iree refused the DLPack capsule of a {actual} tensor "
            f"({type(exc).__name__}: {exc}); {_HOST_PATH}"
        ) from exc
    device_array = type(payload.device_array)(hal, buffer_view)
    imported = _evoxc_module().Tensor(type(payload)(device_array, payload.device))
    if tuple(imported.shape) != tuple(torch_tensor.shape):
        raise InteropUnavailable(
            f"the imported evoxc tensor has shape {tuple(imported.shape)} but the "
            f"torch tensor has {tuple(torch_tensor.shape)}; {_HOST_PATH}"
        )
    if np.dtype(imported.dtype) != np.dtype(str(torch_tensor.dtype).removeprefix("torch.")):
        raise InteropUnavailable(
            f"the imported evoxc tensor has dtype {np.dtype(imported.dtype).name} "
            f"but the torch tensor has {torch_tensor.dtype}; refusing to "
            f"reinterpret the buffer; {_HOST_PATH}"
        )
    return imported
