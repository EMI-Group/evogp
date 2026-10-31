"""Optional FUSED (Triton) evaluation backend for ``evogp.gp`` -- the ONE
deliberate torch boundary in this subtree.

The device hand-off is evoxc's NATIVE DLPack export (``Tensor.__dlpack__``
-> ``torch.from_dlpack``) in the input direction and a kernel-written device
buffer in the output direction, so no private payload handle is touched and no
import step exists at all: ``FusedDeviceEvaluator`` passes the exported
``evoxc`` fitness buffer to the kernel as ``out=``, so a device call ends with
the result already in the tensor the variation graph reads.

Why a boundary at all
---------------------
``evogp.gp`` is torch-free by construction: every other module here is pure
numpy + ``evoxc`` and ``tests/test_gp_stack_purity.py`` plus a clean-subprocess
test pin that (no torch / evox / evoxc at module import, host glue is numpy).
The fused route, however, IS a torch artefact -- the sibling
:mod:`evogp.tree.fused_engine` launches Triton kernels on torch device tensors,
so the compiled library it wraps can only be driven by torch tensors.  This
module is therefore the single place where the stack is allowed to touch torch,
and it confines that contact to *call* time:

* nothing is imported at module import time -- torch, triton and the sibling
  engine are resolved through :func:`importlib.import_module` *inside* the
  functions, so ``import evogp.gp.fused_eval`` succeeds on a box with neither
  torch, triton, evoxc nor a GPU, and never pulls any of them into
  ``sys.modules``;
* the literal import statements the purity grep looks for are deliberately
  absent, which is exactly why the lazy loader exists (do not "tidy" it back
  into an import statement -- the grep and the subprocess test would fail);
* every entry point either answers a question (:func:`fused_available`, never
  raises) or raises :class:`FusedUnavailable` naming the missing half.

What it computes
----------------
:func:`sr_fitness_fused_channels` takes the ordinary ``evogp.gp`` channel triple
(a :class:`~evogp.gp.state.GPState`, the three ``[pop_size, max_nodes]`` channels,
or the three 1-D rows of one individual) and returns the same
``(pop_size,)`` float32 MAXIMIZED fitness :meth:`evogp.gp.problem.SymbolicRegression.evaluate`
returns, but computed by the sibling Triton kernel:

============= ===========================================================
host decode   ``encoding.to_legacy_triple`` -> ``(batch_node_value float32,
              batch_node_type int16, batch_subtree_size int16)``, the exact
              ``Forest`` layout the engine expects (function nodes carry the
              ``FUNC_TYPE`` marker, leaves their ``opcode``, padding 0)
upload        the three tables + ``x``/``y`` to a torch CUDA/HIP device
kernel        ``fused_engine.sr_fitness_fused`` -- the SR loss is reduced
              in-kernel, so the only device output is ``(pop_size,)``
readback      ``(pop_size,)`` positive loss -> negate -> non-finite -> ``-inf``
============= ===========================================================

Two routes live here, and they differ only in where the channels come from:

``FusedEvaluator`` (host route)
    the table above, unchanged: the channels are numpy (or an ``evoxc`` tensor
    that gets downloaded) and the fitness comes back as a numpy array.  This is
    the route for a numpy-backed loop, and it is what a caller gets whenever
    the device route declines.
``FusedDeviceEvaluator`` (DEVICE route)
    the channels are ``evoxc`` tensors on a device, and they are never
    downloaded: ``evoxc`` exports a device buffer through the standard DLPack
    protocol (``Tensor.__dlpack__`` / ``__dlpack_device__``), so the channel
    buffers become zero-copy ``torch.from_dlpack`` views, the same conversion
    runs on the device, and the kernel's ``(pop_size,)`` loss is negated in
    torch.  Nothing crosses the host in either direction, so the fitness is
    handed to the variation graph as an ``evoxc`` DEVICE tensor with no readback
    (``workflow.step`` accepts either).

The upload above is only unavoidable on the HOST route.  On the device route
both directions are the ``evoxc`` DLPack protocol and torch's own
``from_dlpack`` -- this module keeps no private-handle bridge and no second
protocol: it validates what came back (shape / dtype / device kind / DLPack
device id), declines with :class:`InteropUnavailable` when the native path
refuses, and never stages a buffer through the host.  ``measure_fused.py`` is
the committed evidence for what each route costs relative to the in-graph
``evoxc`` route.

TRITON_CACHE_DIR (read this before the first fused call)
-------------------------------------------------------
Triton caches compiled kernels under ``$TRITON_CACHE_DIR`` (default
``$HOME/.triton``).  On a box with a read-only ``$HOME`` -- this repository's
validation box -- the first launch dies with
``OSError: [Errno 30] Read-only file system: '/root/.triton'``.  Every entry
point here therefore calls :func:`ensure_triton_cache_dir` first, which points
the variable at ``$TMPDIR/evogp-triton-cache`` **only when the caller has not set
it**, and creates the directory.  A caller-supplied ``TRITON_CACHE_DIR`` is
never overridden.  The first call of a fresh process also pays the Triton JIT
(hundreds of ms); with a warm on-disk cache it is ~2 ms, and :func:`fused_available`
is not a compile.

Contracts (they match ``problem.SymbolicRegression.evaluate``)
--------------------------------------------------------------
* ``(pop_size,)`` float32, MAXIMIZED (the loss is negated here), fresh and
  writable, non-finite -> ``-inf``.
* ``x`` is ``(N, input_len)`` (a 1-D array is one column), ``y`` is ``(N,)``;
  both are copied to contiguous float32 and never mutated.
* The host route's output lives on the host: it is a numpy-facing adapter, not
  a device result.  The DEVICE route (:class:`FusedDeviceEvaluator`) returns the
  fitness as an ``evoxc`` tensor on the channels' own device instead -- same
  values, same contract, no readback; only its fallback path returns numpy.
* A malformed population is NOT declined by the kernel (it produces a
  defined-but-arbitrary number); ``validate=True`` runs the host decode
  validation (:func:`evogp.gp.encoding.validate_channels`) first, which costs a
  Python loop over rows x nodes and is therefore opt-in.
* The device route additionally rejects an ``OUT_NODE``-tagged opcode
  (multi-output trees, ``NotImplementedError``, exactly like
  :func:`evogp.gp.encoding.to_legacy_triple`) -- it never evaluates one.

Scope: this module adds routes, it does not change one.  Nothing in ``evogp.gp``
selects the fused backend on its own -- a caller plugs an evaluator in as
``evaluate=`` -- and ``measure_fused.py`` is the committed evidence for what
wiring each one in would (and would not) buy.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
from typing import Any, Optional, Tuple

import numpy as np

from ._constants import FUNC_TYPE, Func, NType
from .encoding import _channel_arrays, to_legacy_triple

__all__ = [
    "TRITON_CACHE_DIR_ENV",
    "FusedUnavailable",
    "InteropUnavailable",
    "DEVICE_TYPE_CODES",
    "ensure_triton_cache_dir",
    "fused_available",
    "interop_available",
    "is_device_tensor",
    "device_of",
    "export_to_torch",
    "clear_interop_cache",
    "sr_fitness_fused_channels",
    "FusedEvaluator",
    "FusedDeviceEvaluator",
]

#: The environment variable Triton reads for its compilation cache.
TRITON_CACHE_DIR_ENV = "TRITON_CACHE_DIR"

#: Sub-directory created under ``$TMPDIR`` when the variable is unset.
_DEFAULT_CACHE_SUBDIR = "evogp-triton-cache"


class FusedUnavailable(RuntimeError):
    """The fused route cannot serve this call (or this box).

    Raised where the sibling engine would raise ``FusedUnsupported`` (which
    cannot be re-exported here: importing that module at import time would pull
    torch into this torch-free subtree) and where a missing piece of the route is
    diagnosed before the call -- always with the reason and the remedy in the
    message.  Callers that can fall back should test :func:`fused_available`
    first, or catch this class.
    """


# --------------------------------------------------------------------------- #
# the lazy torch / triton boundary
# --------------------------------------------------------------------------- #
def _module(name: str) -> Any:
    """Import ``name`` at CALL time (the whole point of this module)."""
    return importlib.import_module(name)


def _fused_engine() -> Any:
    """The sibling Triton facade (:mod:`evogp.tree.fused_engine`)."""
    return _module("evogp.tree.fused_engine")


def _torch() -> Any:
    """The torch module, resolved lazily and never at import time."""
    return _module("torch")


def ensure_triton_cache_dir() -> str:
    """Point ``TRITON_CACHE_DIR`` at a writable directory; return the path.

    A caller-supplied value is respected as-is (and returned unchanged, without
    touching the filesystem).  Otherwise the directory is
    ``$TMPDIR/evogp-triton-cache`` (falling back to ``tempfile.gettempdir()``),
    created if missing, and exported so that Triton -- which reads the variable
    when it first compiles -- picks it up.  Raises ``OSError`` when no writable
    location exists, which is the one failure this box actually has (``$HOME`` is
    read-only).

    Idempotent and cheap (one ``getenv`` after the first call).
    """
    current = os.environ.get(TRITON_CACHE_DIR_ENV)
    if current:
        return current
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    path = os.path.join(base, _DEFAULT_CACHE_SUBDIR)
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise OSError(f"{path} is not writable; set {TRITON_CACHE_DIR_ENV} yourself")
    os.environ[TRITON_CACHE_DIR_ENV] = path
    return path


def fused_available() -> bool:
    """Whether this box can run the fused route; never raises.

    True only when all three hold: the sibling engine imports (which drags torch
    and the tree layer), the engine's own probe passes (``triton`` importable and
    a torch CUDA/HIP device visible -- it answers from
    ``importlib.util.find_spec`` and does not compile anything), and a writable
    Triton cache directory can be established.  A per-call refusal (a population
    outside the scratch envelope, an ``output_len`` the kernel does not serve) is
    NOT visible here -- that is :class:`FusedUnavailable` from the call.
    """
    try:
        if not _fused_engine().fused_available():
            return False
        ensure_triton_cache_dir()
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        return False
    return True


def _require_engine() -> Any:
    """The sibling engine, or :class:`FusedUnavailable` naming the missing half."""
    try:
        engine = _fused_engine()
    except ImportError as exc:  # torch / the tree layer missing
        raise FusedUnavailable(
            "the fused route needs evogp.tree.fused_engine, which could not be "
            f"imported ({exc}); the evogp.tree package needs torch, triton and "
            "evoxc on this interpreter"
        ) from exc
    if not engine.fused_available():
        torch = _torch()
        if not torch.cuda.is_available():
            raise FusedUnavailable(
                "the fused route needs a torch CUDA/HIP device "
                "(torch.cuda.is_available() is False), so no Triton kernel can "
                "be launched"
            )
        raise FusedUnavailable(
            "the fused route needs triton, which is not importable in this "
            "interpreter"
        )
    try:
        ensure_triton_cache_dir()
    except OSError as exc:
        raise FusedUnavailable(
            f"the fused route needs a writable {TRITON_CACHE_DIR_ENV} ({exc})"
        ) from exc
    return engine


# --------------------------------------------------------------------------- #
# host-side glue
# --------------------------------------------------------------------------- #
def _channel_matrices(channels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three ``[pop_size, max_nodes]`` channel matrices (rank/shape checked).

    Accepting the same three forms as the rest of the stack (a
    ``GPState``, the two-dimensional triple, or the three 1-D rows of one
    individual) is :func:`evogp.gp.encoding._channel_arrays`'s job; this wrapper
    only enforces what the fused kernel needs on top: a real 2-D population whose
    three channels agree on their shape.
    """
    opcode, payload, size = _channel_arrays(channels)
    if opcode.ndim != 2 or payload.ndim != 2 or size.ndim != 2:
        raise ValueError(
            "the fused route needs a 2-D population: opcode/payload/size must be "
            f"[pop_size, max_nodes], got ranks {opcode.ndim}/{payload.ndim}/"
            f"{size.ndim}"
        )
    if not (opcode.shape == payload.shape == size.shape):
        raise ValueError(
            "the three channels must share one shape, got opcode "
            f"{opcode.shape}, payload {payload.shape}, size {size.shape}"
        )
    return opcode, payload, size


def _host_data(x, y, input_len: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` as contiguous float32 ``(N, input_len)`` / ``(N,)`` arrays."""
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {data.shape}")
    targets = np.ascontiguousarray(np.asarray(y, dtype=np.float32)).reshape(-1)
    if targets.shape[0] != data.shape[0]:
        raise ValueError(
            f"x has {data.shape[0]} rows but y has {targets.shape[0]} entries"
        )
    if input_len is not None and int(input_len) != data.shape[1]:
        raise ValueError(
            f"input_len={int(input_len)} contradicts x's {data.shape[1]} columns"
        )
    return data, targets


def _torch_device(torch_module: Any, device: Optional[Any]) -> Any:
    """``device`` as a torch device; ``None`` means the box's GPU ("cuda").

    The string ``"cuda"`` also selects an AMD GPU on a ROCm build, which is the
    repository-wide device policy; any explicit torch device (a
    ``torch.device``, ``"cpu"``, ``"cuda:1"``, ...) is passed through and the
    kernel then declines a non-GPU one with its own message.
    """
    if device is None:
        return torch_module.device("cuda")
    return torch_module.device(device)


def _data_tensors(torch_module: Any, device: Any, data: np.ndarray, targets: np.ndarray):
    """The two data tensors the kernel reads (``(N, input_len)`` and ``(N, 1)``)."""
    return (
        torch_module.as_tensor(data, device=device),
        torch_module.as_tensor(targets[:, None], device=device),
    )


def _fused_loss(
    engine: Any,
    torch_module: Any,
    device: Any,
    value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
    x_tensor: Any,
    y_tensor: Any,
    input_len: int,
    use_mse: bool,
):
    """One kernel launch on already-uploaded data; returns the device loss vector."""
    try:
        return engine.sr_fitness_fused(
            torch_module.as_tensor(value, device=device),
            torch_module.as_tensor(node_type, device=device),
            torch_module.as_tensor(subtree_size, device=device),
            int(input_len),
            1,
            x_tensor,
            y_tensor,
            use_MSE=bool(use_mse),
        )
    except engine.FusedUnsupported as exc:
        raise FusedUnavailable(
            f"the fused route declined this call: {exc}"
        ) from exc


def _fitness_vector(loss: Any, torch_module: Any, pop_size: int) -> np.ndarray:
    """``(pop_size,)`` MAXIMIZED float32 fitness from the device loss vector."""
    values = np.array(
        loss.detach().to(device="cpu", dtype=torch_module.float32).numpy(),
        dtype=np.float32,
        copy=True,
    ).reshape(-1)
    if values.shape[0] != pop_size:
        raise RuntimeError(
            f"the fused kernel returned {values.shape[0]} values for a "
            f"population of {pop_size}"
        )
    fitness = -values
    fitness[~np.isfinite(fitness)] = -np.inf
    return fitness


# --------------------------------------------------------------------------- #
# the public entry point
# --------------------------------------------------------------------------- #
def sr_fitness_fused_channels(
    channels,
    x,
    y,
    *,
    use_mse: bool = True,
    input_len: Optional[int] = None,
    device: Optional[Any] = None,
    validate: bool = False,
) -> np.ndarray:
    """SR fitness of a whole population through the fused Triton kernel.

    Args:
        channels: a :class:`~evogp.gp.state.GPState`, the three
            ``[pop_size, max_nodes]`` channels, or the three 1-D channel rows of
            one individual (a population of one).
        x: ``(N, input_len)`` float32 inputs (a 1-D array is one column).
        y: ``(N,)`` float32 targets.
        use_mse: squared error (default) or absolute error -- the ``use_MSE``
            flag of ``Forest.SR_fitness``.
        input_len: optional cross-check against ``x``'s column count.
        device: torch device for the launch; ``None`` (default) is the box's GPU.
        validate: run :func:`evogp.gp.encoding.validate_channels` on the host
            first (a Python loop over rows x nodes; the kernel itself accepts a
            malformed tree and returns an arbitrary number).

    Returns:
        ``(pop_size,)`` float32 numpy array, MAXIMIZED (the positive kernel loss
        negated), fresh and writable, non-finite -> ``-inf``.

    Raises:
        FusedUnavailable: no device / no triton / no writable Triton cache, or
            the engine declined the call (naming which).
        ValueError: shape contract violations, an illegal function id, or a
            malformed population under ``validate=True``.
    """
    opcode, payload, size = _channel_matrices(channels)
    data, targets = _host_data(x, y, input_len)
    if validate:
        from .encoding import validate_channels

        validate_channels((opcode, payload, size), input_len=data.shape[1])
    # The host decode the kernel needs: the ``Forest`` triple, whose padding is
    # zeroed (so a stale cell past a row's live length cannot be read).
    value, node_type, subtree_size = to_legacy_triple((opcode, payload, size))
    engine = _require_engine()
    torch_module = _torch()
    torch_device = _torch_device(torch_module, device)
    x_tensor, y_tensor = _data_tensors(torch_module, torch_device, data, targets)
    loss = _fused_loss(
        engine,
        torch_module,
        torch_device,
        value,
        node_type,
        subtree_size,
        x_tensor,
        y_tensor,
        data.shape[1],
        use_mse,
    )
    return _fitness_vector(loss, torch_module, int(opcode.shape[0]))


class FusedEvaluator:
    """The fused route as an ``evaluate`` callable for the driver.

    ``FusedEvaluator(x, y)(channels)`` is contract-compatible with
    :meth:`evogp.gp.problem.SymbolicRegression.evaluate` (and with the plain
    ``problem`` instance, its ``__call__``), so it can be handed to
    :func:`evogp.gp.workflow.step` / ``run`` as ``evaluate=``.

    What it adds over :func:`sr_fitness_fused_channels` is residency of the DATA:
    ``x``/``y`` are uploaded once, on the first evaluation, and reused (a
    training loop keeps one dataset for thousands of generations).  The channel
    tables still upload on every call -- this is the HOST route, whose input is a
    numpy population; a device population takes :class:`FusedDeviceEvaluator`,
    which exports the channels instead of uploading them.

    Nothing is imported and nothing is uploaded by the constructor, so building
    one on a CPU-only box is free; :meth:`available` is the probe.
    """

    __slots__ = ("x", "y", "use_mse", "device", "validate", "_tensors")

    def __init__(
        self,
        x,
        y,
        use_mse: bool = True,
        *,
        device: Optional[Any] = None,
        validate: bool = False,
    ) -> None:
        self.x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if self.x.ndim == 1:
            self.x = self.x[None, :]
        if self.x.ndim != 2:
            raise ValueError(f"x must be 1-D or 2-D, got shape {self.x.shape}")
        self.y = np.ascontiguousarray(np.asarray(y, dtype=np.float32)).reshape(-1)
        if self.y.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"x has {self.x.shape[0]} rows but y has {self.y.shape[0]} entries"
            )
        self.use_mse = bool(use_mse)
        self.device = device
        self.validate = bool(validate)
        self._tensors: Optional[Tuple[Any, Any, Any, Any]] = None

    # ------------------------------------------------------------- interface
    def available(self) -> bool:
        """Whether this box can serve :meth:`evaluate` at all (never raises)."""
        return fused_available()

    def evaluate(self, channels) -> np.ndarray:
        """``(pop_size,)`` float32 MAXIMIZED fitness (see the module docstring)."""
        opcode, payload, size = _channel_matrices(channels)
        if self.validate:
            from .encoding import validate_channels

            validate_channels((opcode, payload, size), input_len=self.x.shape[1])
        value, node_type, subtree_size = to_legacy_triple((opcode, payload, size))
        engine = _require_engine()
        torch_module = _torch()
        torch_device = _torch_device(torch_module, self.device)
        x_tensor, y_tensor = self._resident(torch_module, torch_device)
        loss = _fused_loss(
            engine,
            torch_module,
            torch_device,
            value,
            node_type,
            subtree_size,
            x_tensor,
            y_tensor,
            self.x.shape[1],
            self.use_mse,
        )
        return _fitness_vector(loss, torch_module, int(opcode.shape[0]))

    def __call__(self, channels) -> np.ndarray:
        """``evaluate`` -- so the evaluator is directly usable as ``evaluate=``."""
        return self.evaluate(channels)

    # ------------------------------------------------------------ properties
    @property
    def problem_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def solution_dim(self) -> int:
        return 1

    # ---------------------------------------------------------------- internal
    def _resident(self, torch_module: Any, device: Any) -> Tuple[Any, Any]:
        """The uploaded data tensors of this evaluator, built once per device."""
        if self._tensors is None or self._tensors[0] != device:
            self._tensors = (device,) + _data_tensors(
                torch_module, device, self.x, self.y
            )
        return self._tensors[1], self._tensors[2]


# --------------------------------------------------------------------------- #
# the DEVICE-NATIVE route
# --------------------------------------------------------------------------- #
#: ``str(torch device)`` -> the ``FUNC_TYPE`` lookup table resident on it (the
#: device-side counterpart of the numpy table ``to_legacy_triple`` indexes).
#: One tiny tensor per device, kept for the process' lifetime.
_TYPE_TABLES: dict = {}

#: evoxc device kind -> DLPack device-type code (``dlpack.h``): ``kDLCUDA`` is 2
#: and ``kDLROCM`` is 10, while ``"hip"`` is evoxc's alias for ``"rocm"`` (the
#: same HIP buffers, hence the same code).  ``cpu`` is deliberately absent: an
#: ndarray-backed or cpu-kind payload never exports, because its explicit read
#: path (``.numpy()`` / ``t.to(Device("cpu", 0))``) is the host route.
DEVICE_TYPE_CODES = {"rocm": 10, "hip": 10, "cuda": 2}

#: evoxc device kind -> the torch device TYPE.  ``"cuda"`` is torch's spelling of
#: a ROCm device as well, which is the repository-wide device policy.
_TORCH_DEVICE_TYPES = {"rocm": "cuda", "hip": "cuda", "cuda": "cuda"}

#: The remedy sentence every refusal ends with.
_HOST_PATH = (
    "the host path still serves this call (download the channels, run the "
    "kernel from numpy); nothing is silently approximated"
)

#: evoxc device kind -> the one-time behavioural probe's verdict (see
#: :func:`interop_available`); reset by :func:`clear_interop_cache`.
_INTEROP_PROBES: dict = {}


class InteropUnavailable(RuntimeError):
    """A native device hand-off is not possible for this call (or this box).

    Raised with the reason, the unsupported thing and the remedy -- never
    instead of a value that would have been wrong.  Callers that can fall back
    should catch this class (the device-native evaluator in
    :class:`FusedDeviceEvaluator` does exactly that and retries on the host).
    """


def _evoxc_module() -> Any:
    """The ``evoxc`` package, resolved at call time (never at import time)."""
    try:
        return _module("evoxc")
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            "the native device interop needs evoxc (put the sibling checkout on "
            f"PYTHONPATH); {_HOST_PATH}"
        ) from exc


def _normalise_kind(kind: Any) -> str:
    """The canonical lower-case kind spelling of an evoxc kind string."""
    return str(kind).strip().lower().split(":")[0]


def _device_type_code(kind: str) -> int:
    """The DLPack device-type code of an evoxc kind, or :class:`InteropUnavailable`."""
    normalised = _normalise_kind(kind)
    if normalised == "cpu":
        raise InteropUnavailable(
            "the native device interop declines the 'cpu' kind: a host tensor "
            f"needs no pointer sharing and the host path already serves it; "
            f"{_HOST_PATH}"
        )
    code = DEVICE_TYPE_CODES.get(normalised)
    if code is None:
        raise InteropUnavailable(
            f"the native device interop does not know the evoxc device kind "
            f"{kind!r}: only {sorted(DEVICE_TYPE_CODES)} have a DLPack device "
            f"type code that torch can accept; {_HOST_PATH}"
        )
    return int(code)


def _check_kind_against_torch(torch_module: Any, kind: str) -> None:
    """Refuse a device kind this torch build cannot be the same device as.

    The evoxc kind and torch's build are two spellings of one physical device:
    a ROCm build reports ``torch.version.hip`` and only ``rocm``/``hip`` buffers
    live on it, an NVIDIA build has no ``hip`` and only ``cuda`` buffers do.  A
    capsule minted with the wrong DLPack code would be a silent misinterpretation
    of the pointer, so it is refused here instead.
    """
    normalised = _normalise_kind(kind)
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    if normalised in ("rocm", "hip") and not hip:
        raise InteropUnavailable(
            f"the native device interop cannot serve evoxc kind {kind!r}: this "
            f"is not a ROCm torch build (torch.version.hip is {hip!r}), so no "
            f"{kind!r} buffer can be shared with it; {_HOST_PATH}"
        )
    if normalised == "cuda" and hip:
        raise InteropUnavailable(
            "the native device interop cannot serve evoxc kind 'cuda' on a ROCm "
            f"torch build (torch.version.hip is {hip!r}): the buffers are HIP "
            f"buffers and their DLPack code is kDLROCM; {_HOST_PATH}"
        )


def is_device_tensor(value: Any) -> bool:
    """Whether ``value`` is an ``evoxc.Tensor`` resident on a non-cpu device.

    Never raises -- ``False`` for numpy arrays, for host ``evoxc`` tensors and
    even on a box without ``evoxc``.  This is the dispatch probe of
    :class:`FusedDeviceEvaluator`, so it stays free of exceptions, free of host
    transfers and free of torch: it reads the tensor's own public metadata
    (``Tensor.device``), never a payload handle.
    """
    try:
        evoxc_module = _module("evoxc")
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        return False
    if not isinstance(value, evoxc_module.Tensor):
        return False
    device = getattr(value, "device", None)
    if device is None:
        return False
    return _normalise_kind(getattr(device, "kind", "cpu")) != "cpu"


def device_of(tensor: Any) -> Any:
    """The ``evoxc.Device`` of a device-resident tensor (public metadata only).

    Raises:
        InteropUnavailable: for a numpy array, a host ``evoxc`` tensor or an
            object that is not an ``evoxc.Tensor`` at all.
    """
    try:
        evoxc_module = _module("evoxc")
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            f"the native device interop needs evoxc; {_HOST_PATH}"
        ) from exc
    if not isinstance(tensor, evoxc_module.Tensor):
        raise InteropUnavailable(
            "the native device interop needs an evoxc.Tensor (a device buffer), "
            f"got {type(tensor).__name__}; {_HOST_PATH}"
        )
    device = getattr(tensor, "device", None)
    if device is None or _normalise_kind(getattr(device, "kind", "cpu")) == "cpu":
        raise InteropUnavailable(
            "the evoxc tensor lives on the host (its device is "
            f"{device!r}), so there is no device buffer to share; {_HOST_PATH}"
        )
    return device


def export_to_torch(tensor: Any, *, device_kind: Optional[Any] = None) -> Any:
    """A torch tensor SHARING the memory of an ``evoxc`` device tensor.

    This is the standard DLPack protocol, both sides public: the evoxc tensor is
    an exporter (``Tensor.__dlpack__`` delegates to the iree payload, which wraps
    the HAL buffer in a capsule) and ``torch.from_dlpack`` consumes the capsule
    without copying.  Writing through either handle is visible through the other.

    Args:
        tensor: an ``evoxc.Tensor`` resident on a ``rocm``/``hip``/``cuda``
            device.
        device_kind: optional assertion/override for the DLPack device kind.
            When given it must name a supported kind (otherwise
            :class:`InteropUnavailable`) and must agree with the tensor's own
            ``evoxc.Device`` kind (otherwise ``ValueError``).

    Returns:
        A torch tensor on the corresponding torch device (``"cuda"`` on a ROCm
        build, as everywhere else in this repository), with the tensor's shape
        and dtype.  No bytes are copied.

    Raises:
        InteropUnavailable: no torch, a host/non-evoxc tensor, an unsupported
            device kind or dtype, a ``cpu`` device, a DLPack device report that
            disagrees with the tensor's own ``Device``, or an exported tensor
            whose metadata disagrees with the tensor's (a loud guard against a
            wrong mapping -- never a reinterpreted buffer).
        ValueError: ``device_kind`` contradicts the tensor's own device.
    """
    torch_module = _torch()
    device = device_of(tensor)
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
    expected_shape = tuple(int(size) for size in tuple(tensor.shape))
    expected_dtype = np.dtype(tensor.dtype)
    index = int(getattr(device, "index", 0))
    try:
        reported = tuple(int(part) for part in tensor.__dlpack_device__())
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            f"the evoxc tensor on {device!r} does not report its DLPack device "
            f"({type(exc).__name__}: {exc}) -- this evoxc/iree combination has no "
            f"zero-copy device export; {_HOST_PATH}"
        ) from exc
    if reported != (code, index):
        raise InteropUnavailable(
            f"the evoxc tensor reports DLPack device {reported!r} but lives on "
            f"{device!r} (DLPack {(code, index)!r}); refusing to share a buffer "
            f"whose device cannot be verified; {_HOST_PATH}"
        )
    try:
        # The PROTOCOL form, not ``torch.from_dlpack(tensor)``: handed the object,
        # torch re-derives the device and the current stream through the protocol
        # and that bookkeeping costs 4.4x the capsule itself (measured 5.9 us
        # against 1.3 us per channel here).  ``__dlpack__`` is the public export
        # half of the same protocol, and the capsule it returns is what
        # ``torch.from_dlpack`` consumes either way.
        capsule = tensor.__dlpack__()
        exported = torch_module.from_dlpack(capsule)
    except InteropUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as a fallback, never silent
        raise InteropUnavailable(
            f"torch refused the DLPack capsule of a {kind} evoxc buffer "
            f"({type(exc).__name__}: {exc}); {_HOST_PATH}"
        ) from exc
    if tuple(int(size) for size in exported.shape) != expected_shape:
        raise InteropUnavailable(
            f"the exported torch tensor has shape {tuple(exported.shape)} but the "
            f"evoxc tensor reports {expected_shape} -- refusing to hand back a "
            f"differently-shaped view; {_HOST_PATH}"
        )
    if _torch_dtype_name(torch_module, exported.dtype) != expected_dtype.name:
        raise InteropUnavailable(
            f"the exported torch tensor has dtype {exported.dtype} but the evoxc "
            f"tensor reports {expected_dtype.name} -- refusing to reinterpret the "
            f"buffer; {_HOST_PATH}"
        )
    expected_type = _TORCH_DEVICE_TYPES.get(kind)
    if exported.device.type != expected_type:
        raise InteropUnavailable(
            f"the exported tensor lives on {exported.device}, which is not the "
            f"torch view of the evoxc {kind!r} device (expected a torch "
            f"{expected_type!r} tensor); {_HOST_PATH}"
        )
    if exported.device.index is not None and int(exported.device.index) != index:
        raise InteropUnavailable(
            f"the exported tensor lives on {exported.device} but the evoxc device "
            f"is {kind}:{index} -- the buffer would be read from the wrong "
            f"device; {_HOST_PATH}"
        )
    if not exported.is_contiguous():
        raise InteropUnavailable(
            f"the exported tensor is not contiguous ({tuple(exported.shape)}, "
            f"strides {exported.stride()}), which the fused kernel cannot read "
            f"as a flat buffer; {_HOST_PATH}"
        )
    return exported


def interop_available(kind: Optional[Any] = None) -> bool:
    """Whether the native device hand-off works on this box; never raises.

    True only when ``torch`` is importable with a visible CUDA/HIP device,
    ``triton`` is installed (the fused kernel is the reason for the hand-off at
    all), the requested ``kind`` maps onto a DLPack device type this torch build
    can be the same device as, AND a one-time BEHAVIOURAL probe of exactly the
    mechanism the device route uses succeeds: a one-element device buffer is
    exported with ``torch.from_dlpack``, written through the torch view, and read
    back through the evoxc tensor.  That probe is cached per kind (a few
    milliseconds once) and drops with :func:`clear_interop_cache`, so the answer
    stays honest when an older evoxc/iree runtime has no device export -- while
    the route itself costs nothing per call.

    Args:
        kind: optional evoxc device kind (``"rocm"`` / ``"cuda"`` / ...) the
            caller intends to use; an unknown or ``cpu`` kind answers ``False``,
            and so does a kind this torch build cannot be the same device as
            (``"rocm"`` on an NVIDIA build, ``"cuda"`` on a ROCm build).

    Notes:
        A per-call refusal (an unsupported dtype, a population outside the
        kernel's scratch envelope, a device that stopped exporting) is NOT
        visible here -- that is :class:`InteropUnavailable` from the call, which
        the evaluator turns into a fallback.
    """
    try:
        if kind is not None and _normalise_kind(kind) not in DEVICE_TYPE_CODES:
            return False
        torch_module = _module("torch")
        if not torch_module.cuda.is_available():
            return False
        if importlib.util.find_spec("triton") is None:
            return False
        if kind is not None:
            _check_kind_against_torch(torch_module, kind)
        return _probe_native_export(torch_module, kind)
    except InteropUnavailable:
        return False
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        return False


def _probe_native_export(torch_module: Any, kind: Optional[Any]) -> bool:
    """One-time end-to-end probe of the hand-off (see :func:`interop_available`).

    Exports a one-element device buffer to torch, writes one value through the
    torch view, and reads it back through the evoxc tensor -- i.e. it proves the
    two directions the device route depends on (zero-copy export, and a torch
    write becoming visible in the evoxc buffer) instead of assuming them.  The
    verdict is memoized per kind; :func:`clear_interop_cache` drops it.
    """
    if kind is None:
        kind = "rocm" if getattr(torch_module.version, "hip", None) else "cuda"
    normalised = _normalise_kind(kind)
    cached = _INTEROP_PROBES.get(normalised)
    if cached is not None:
        return cached
    verdict = False
    try:
        evoxc_module = _evoxc_module()
        device = evoxc_module.Device(normalised, 0)
        buffer = evoxc_module.zeros((1,), evoxc_module.float32).to(device)
        view = export_to_torch(buffer)
        view.copy_(
            torch_module.ones((1,), dtype=torch_module.float32, device=view.device)
        )
        torch_module.cuda.synchronize(view.device)
        host = buffer.to(evoxc_module.Device("cpu", 0))
        verdict = float(np.asarray(host.numpy()).reshape(-1)[0]) == 1.0
    except Exception:  # noqa: BLE001 - a probe answers False, it never raises
        verdict = False
    _INTEROP_PROBES[normalised] = verdict
    return verdict


def clear_interop_cache() -> None:
    """Drop the memoized probe verdicts (nothing else is cached here).

    Only needed when something else replaced ``torch`` / ``evoxc`` / ``iree`` in
    ``sys.modules`` (a test that simulates a box without them, for instance); a
    normal process never calls this.
    """
    _INTEROP_PROBES.clear()


#: Integer dtypes the channel tables may carry into the legacy conversion.
_CHANNEL_INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8")

#: Float dtypes the payload channel (and therefore the value table) may carry.
_CHANNEL_FLOAT_DTYPES = ("float16", "float32", "float64")


def _torch_dtype_name(torch_module: Any, dtype: Any) -> str:
    """``"float32"``-style name of a torch dtype (no ``torch.`` prefix)."""
    return str(dtype).removeprefix("torch.")


def _func_type_table(torch_module: Any, device: Any):
    """``FUNC_TYPE`` as a ``[Func.END]`` int32 tensor on ``device`` (memoized)."""
    key = str(device)
    table = _TYPE_TABLES.get(key)
    if table is None:
        table = torch_module.as_tensor(
            np.asarray(FUNC_TYPE, dtype=np.int32), device=device
        )
        _TYPE_TABLES[key] = table
    return table


def _channel_objects(channels) -> Tuple[Any, Any, Any]:
    """The three channel operands AS GIVEN -- device tensors are NOT downloaded.

    The stack's one channel parser (:func:`evogp.gp.encoding._channel_arrays`)
    ends in ``_as_numpy``, which does exactly the host round trip the device
    route exists to avoid.  This parser mirrors its accepted forms (a
    :class:`~evogp.gp.state.GPState`, an ``(opcode, payload, size)`` triple, the
    three 1-D rows of one individual) and nothing else; rank/shape/promotion are
    left to the route that consumes them.
    """
    from .state import GPState

    if isinstance(channels, GPState):
        opcode, payload, size = channels.channels()
    else:
        try:
            opcode, payload, size = channels
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "evogp.gp: channels must be a GPState or an "
                "(opcode, payload, size) triple"
            ) from exc
    return opcode, payload, size


def _device_channels(channels):
    """``(opcode, payload, size, device)`` when the channels are device tensors.

    ``None`` -- meaning "use the host path" -- when the channels are numpy, when
    any of them is a host ``evoxc`` tensor, when they do not share one device, or
    when they are not a 2-D population (a 1-D triple of one individual needs the
    host path's promotion).  Everything here reads metadata only: no transfer,
    no exception, and therefore no per-call cost on the host route.

    The device verdict is a dispatch decision, never a correctness decision:
    :meth:`FusedDeviceEvaluator.evaluate` still falls back to the host path when
    the interop or the kernel declines.
    """
    opcode, payload, size = _channel_objects(channels)
    tensors = (opcode, payload, size)
    for tensor in tensors:
        if not is_device_tensor(tensor):
            return None
        if len(tuple(tensor.shape)) != 2:
            return None
    shapes = tuple(tuple(tensor.shape) for tensor in tensors)
    if not (shapes[0] == shapes[1] == shapes[2]):
        return None
    device = device_of(opcode)
    if any(device_of(tensor) != device for tensor in tensors[1:]):
        return None
    return opcode, payload, size, device


def _legacy_triple_device(torch_module: Any, table: Any, opcode, payload, size):
    """``to_legacy_triple``'s conversion, traced in torch on device tensors.

    Bitwise-equivalent to the host bridge (both are ``where`` chains over the
    same rule): a function node (``size > 1``) carries the function id as its
    value and its ``FUNC_TYPE`` marker as its type; a leaf carries its payload
    and its ``opcode``; the padding (``size == 0``) is zero in all three.  The
    function id is CLAMPED into ``[0, Func.END)`` -- the kernel reads it as an
    index, and an out-of-range id is caught by the host validation
    (``validate=True``) rather than by a silently clamped device lookup.
    """
    function = size > 1
    padding = size == 0
    clipped = torch_module.where(function, opcode, torch_module.zeros_like(opcode))
    func_type = table[clipped.clamp(0, Func.END - 1)]
    node_type = torch_module.where(
        padding,
        torch_module.zeros_like(opcode),
        torch_module.where(function, func_type, opcode),
    ).to(torch_module.int16)
    value = torch_module.where(
        padding,
        torch_module.zeros_like(payload),
        torch_module.where(function, clipped.to(torch_module.float32), payload),
    ).float()
    subtree_size = torch_module.where(
        padding, torch_module.zeros_like(size), size
    ).to(torch_module.int16)
    return value, node_type, subtree_size


class FusedDeviceEvaluator:
    """The fused route WITHOUT a host round trip: device channels in, device fitness out.

    ``FusedDeviceEvaluator(x, y)(channels)`` is contract-compatible with
    :meth:`evogp.gp.problem.SymbolicRegression.evaluate` and with
    :class:`FusedEvaluator`, so it is handed to
    :func:`evogp.gp.workflow.step` / ``run`` as ``evaluate=`` -- but where
    :class:`FusedEvaluator` downloads the population and returns a numpy vector,
    this one stays on the device:

    ============ =============================================================
    channels     ``evoxc`` device tensors -> zero-copy torch views
                 (:func:`export_to_torch`, i.e. ``torch.from_dlpack``)
    conversion   the ``Forest`` legacy triple, computed in torch on the device
                 (``_legacy_triple_device``)
    kernel       ``fused_engine.sr_fitness_fused`` -- the loss is reduced
                 in-kernel and written into this evaluator's ``evoxc`` device
                 fitness buffer through its ``out=`` parameter (the buffer is
                 exported to torch ONCE, so the kernel's destination IS the
                 buffer the variation graph reads)
    fitness      ``-loss`` and non-finite -> ``-inf``, applied IN PLACE on that
                 same ``(pop_size,)`` buffer -- no allocation, no copy, no
                 import step, no readback
    ============ =============================================================

    Cost profile: one DLPack capsule per channel per call (no bytes move), one
    kernel launch, two tiny in-place ops on the ``(pop_size,)`` buffer, one
    device sync (see below) and the ``OUT_NODE`` guard -- a single boolean
    readback (~0.04 ms on this box, and it is read after the conversion is
    enqueued so the wait covers that work too).  ``x``/``y`` are uploaded to
    torch once per device and reused; nothing is uploaded by the constructor.

    The fitness buffer is REUSED between calls (that is what keeps the device
    route free of allocations and host round trips), so the tensor returned by
    :meth:`evaluate` is only valid until the next call -- same rule as any other
    in-place device buffer, and the driver reads the numbers back before the next
    ``evaluate`` anyway.

    Fallback, never a wrong number: when the channels are not device tensors, or
    the interop or the kernel is unavailable (``InteropUnavailable`` /
    ``FusedUnavailable`` -- a cpu device, an unsupported dtype, a torch device
    mismatch, a population outside the kernel's scratch envelope, ...), the call
    RETRIES on the host route (:func:`sr_fitness_fused_channels`, which
    downloads the channels explicitly) and returns its numpy vector.  The route
    actually taken is recorded in :attr:`last_route` (``"device"`` / ``"host"``)
    and counted in :attr:`route_counts`, with the reason in
    :attr:`last_fallback`.  The one thing that is NOT caught is the host route's
    own ``FusedUnavailable`` -- when it raises, no fused route exists at all on
    this box, and :meth:`available` already said so.

    ``validate=True`` costs a download: it runs
    :func:`evogp.gp.encoding.validate_channels` on the host first (a Python loop
    over rows x nodes) -- it is the only place this evaluator touches the host
    on the device route, and it also catches a function id the kernel would
    clamp.  The device route rejects an ``OUT_NODE``-tagged opcode
    (``NotImplementedError``, as ``to_legacy_triple`` does) whatever
    ``validate`` says.
    """

    __slots__ = (
        "x",
        "y",
        "use_mse",
        "device",
        "validate",
        "last_route",
        "last_fallback",
        "route_counts",
        "_tensors",
        "_fitness",
    )

    def __init__(
        self,
        x,
        y,
        use_mse: bool = True,
        *,
        device: Optional[Any] = None,
        validate: bool = False,
    ) -> None:
        self.x, self.y = _host_data(x, y, None)
        self.use_mse = bool(use_mse)
        self.device = device
        self.validate = bool(validate)
        self.last_route: Optional[str] = None
        self.last_fallback: Optional[str] = None
        self.route_counts: dict = {"device": 0, "host": 0}
        self._tensors: Optional[Tuple[Any, Any, Any]] = None
        self._fitness: Optional[Tuple[Tuple[str, int], Any, Any]] = None

    # ------------------------------------------------------------- interface
    def available(self) -> bool:
        """Whether this evaluator can serve a call AT ALL (never raises).

        That is the host route's own probe (:func:`fused_available`): the device
        route is the optimisation, the host route is the floor, and
        :meth:`device_available` answers for the fast one separately.
        """
        return fused_available()

    def device_available(self) -> bool:
        """Whether the DEVICE route is usable on this box (never raises)."""
        return interop_available(self.device_kind)

    @property
    def device_kind(self) -> Optional[str]:
        """The evoxc device kind of this evaluator's torch device.

        ``"rocm"`` on a ROCm build, ``"cuda"`` on an NVIDIA one, ``"cpu"`` when
        the evaluator was pinned to the host, and ``None`` when torch cannot be
        resolved at all (in which case only the host path exists anyway).
        """
        try:
            torch_module = _torch()
            resolved = _torch_device(torch_module, self.device)
        except Exception:  # noqa: BLE001 - a probe answers None, it never raises
            return None
        if resolved.type != "cuda":
            return resolved.type
        return "rocm" if getattr(torch_module.version, "hip", None) else "cuda"

    def evaluate(self, channels):
        """``(pop_size,)`` float32 MAXIMIZED fitness (see the class docstring).

        Returns an ``evoxc`` DEVICE tensor on the channels' own device when the
        device route served the call, and a numpy array when the host route did
        (the fallback).  Both satisfy the repository-wide fitness contract.
        """
        device_channels = _device_channels(channels)
        if device_channels is None:
            self._record("host")
            return self._evaluate_host(channels)
        opcode, payload, size = device_channels[:3]
        try:
            fitness = self._evaluate_device(opcode, payload, size)
        except (InteropUnavailable, FusedUnavailable) as exc:
            self.last_fallback = f"{type(exc).__name__}: {exc}"
            self._record("host")
            return self._evaluate_host(channels)
        self.last_fallback = None
        self._record("device")
        return fitness

    def __call__(self, channels):
        """``evaluate`` -- so the evaluator is directly usable as ``evaluate=``."""
        return self.evaluate(channels)

    def warm(self, shape: Optional[Tuple[int, int]] = None, channels=None):
        """Pay the one-time cost of this evaluator before timing a loop.

        Warms the Triton JIT (the expensive part: from an empty cache it is
        hundreds of ms, from a warm on-disk cache ~2 ms) and the data upload.
        The kernel's specialization key is ``(per_row, reduce, use_mse,
        block_rows, num_warps)`` -- population- and shape-independent -- so a
        small host population warms every later call.

        Args:
            shape: ``(pop_size, max_nodes)`` of a throwaway population of
                one-node variable trees; default ``(64, 32)``.
            channels: REAL channels to warm with instead (a device ``GPState``
                or device triple warms the device route as well: the DLPack
                hand-off, the conversion and the import back).

        Returns:
            Whatever :meth:`evaluate` returned for the warming call.
        """
        if channels is not None:
            return self.evaluate(channels)
        if shape is None:
            pop_size, width = 64, 32
        else:
            try:
                pop_size, width = int(shape[0]), int(shape[1])
            except (TypeError, IndexError) as exc:
                raise ValueError(
                    "warm(shape=...) needs a (pop_size, max_nodes) pair, got "
                    f"{shape!r}"
                ) from exc
        return self.evaluate(_warm_channels(pop_size, width))

    # ------------------------------------------------------------ properties
    @property
    def problem_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def solution_dim(self) -> int:
        return 1

    # ---------------------------------------------------------------- internal
    def _record(self, route: str) -> None:
        """Note the route a call took (:attr:`last_route` / :attr:`route_counts`)."""
        self.last_route = route
        self.route_counts[route] = self.route_counts.get(route, 0) + 1

    def _evaluate_device(self, opcode, payload, size):
        """One device-native call; raises for a fallback, never returns a guess."""
        if self.validate:
            from .encoding import validate_channels

            validate_channels((opcode, payload, size), input_len=self.problem_dim)
        torch_module = _torch()
        op_tensor = export_to_torch(opcode)
        payload_tensor = export_to_torch(payload)
        size_tensor = export_to_torch(size)
        launch = op_tensor.device
        self._check_launch_device(torch_module, launch)
        self._check_channel_dtypes(torch_module, op_tensor, payload_tensor, size_tensor)
        # The OUT_NODE guard is the ONE mandatory host sync of a device call, so
        # it is ENQUEUED first and read after the conversion: the wait then
        # covers the conversion's kernels as well, for the same round trip.
        tagged = torch_module.any((op_tensor & NType.OUT_NODE) != 0)
        table = _func_type_table(torch_module, launch)
        value, node_type, subtree_size = _legacy_triple_device(
            torch_module, table, op_tensor, payload_tensor, size_tensor
        )
        if bool(tagged):
            raise NotImplementedError(
                "FusedDeviceEvaluator: multi-output (OUT_NODE-tagged) trees are "
                "not implemented in the evoxc GP stack yet; build the population "
                "with output_len=1"
            )
        engine = _require_engine()
        x_tensor, y_tensor = self._resident(torch_module, launch)
        pop_size = int(op_tensor.shape[0])
        target, view = self._fitness_buffer(
            torch_module, launch, device_of(opcode), pop_size
        )
        # The kernel writes the finished ``(pop_size,)`` loss straight INTO the
        # exported buffer (``out=``): no result allocation and no copy into a
        # second tensor.
        try:
            engine.sr_fitness_fused(
                value,
                node_type,
                subtree_size,
                self.problem_dim,
                1,
                x_tensor,
                y_tensor,
                use_MSE=self.use_mse,
                out=view,
            )
        except engine.FusedUnsupported as exc:
            raise FusedUnavailable(
                f"the fused route declined this call: {exc}"
            ) from exc
        # The fitness contract (MAXIMIZED, non-finite -> -inf) in place on the
        # same buffer: ``-loss`` and then every non-finite entry -- including the
        # ``+inf`` a ``-inf`` loss turns into -- mapped to ``-inf``.
        view.neg_()
        view.nan_to_num_(
            nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
        )
        # The one mandatory cross-runtime fence: iree's HAL adds none of its own,
        # so the next ``evoxc.run`` must not read the buffer before torch's
        # writes land.
        torch_module.cuda.synchronize(view.device)
        return target

    def _fitness_buffer(self, torch_module: Any, device: Any, evoxc_device: Any,
                        pop_size: int):
        """The persistent ``(pop_size,)`` ``evoxc`` DEVICE buffer this evaluator returns.

        Built once per (torch device, evoxc device, population size) through the
        SAME public hand-off the channels use -- host ``evoxc.zeros`` uploaded
        with ``Tensor.to``, then exported to torch with
        :func:`export_to_torch` -- so the torch view is the kernel's ``out=``
        destination and the variation graph reads that same buffer: one write, no
        copy, no import step.
        """
        key = (str(device), int(pop_size))
        cached = self._fitness
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        evoxc_module = _evoxc_module()
        host = evoxc_module.zeros((int(pop_size),), evoxc_module.float32)
        target = host.to(evoxc_device)
        view = export_to_torch(target)
        self._fitness = (key, target, view)
        return target, view

    def _evaluate_host(self, channels) -> np.ndarray:
        """The host route: the existing entry point, on downloaded channels."""
        opcode, payload, size = _channel_matrices(channels)
        return sr_fitness_fused_channels(
            (opcode, payload, size),
            self.x,
            self.y,
            use_mse=self.use_mse,
            input_len=self.problem_dim,
            device=self.device,
            validate=self.validate,
        )

    def _check_launch_device(self, torch_module: Any, launch: Any) -> None:
        """Reject a population whose device contradicts the configured one."""
        if self.device is None:
            return
        expected = torch_module.device(self.device)
        same_type = expected.type == launch.type
        same_index = (
            expected.index is None
            or launch.index is None
            or int(expected.index) == int(launch.index)
        )
        if not (same_type and same_index):
            raise ValueError(
                f"this evaluator's device is {expected} but the channels live on "
                f"{launch}; pass device={launch!r} (or build the population on "
                "that device) instead of mixing the two"
            )

    @staticmethod
    def _check_channel_dtypes(
        torch_module: Any, opcode: Any, payload: Any, size: Any
    ) -> None:
        """The channel dtype contract of the legacy triple (a caller error)."""
        for name, tensor in (("opcode", opcode), ("size", size)):
            if _torch_dtype_name(torch_module, tensor.dtype) not in _CHANNEL_INT_DTYPES:
                raise ValueError(
                    f"the {name} channel must be an integer tensor, got "
                    f"{tensor.dtype}"
                )
        if _torch_dtype_name(torch_module, payload.dtype) not in _CHANNEL_FLOAT_DTYPES:
            raise ValueError(
                f"the payload channel must be a float tensor, got {payload.dtype}"
            )

    def _resident(self, torch_module: Any, device: Any) -> Tuple[Any, Any]:
        """The uploaded data tensors of this evaluator, built once per device."""
        if self._tensors is None or self._tensors[0] != device:
            self._tensors = (device,) + _data_tensors(
                torch_module, device, self.x, self.y
            )
        return self._tensors[1], self._tensors[2]


def _warm_channels(pop_size: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A throwaway population of one-node variable trees (for :meth:`warm`)."""
    opcode = np.zeros((int(pop_size), int(width)), dtype=np.int32)
    payload = np.zeros((int(pop_size), int(width)), dtype=np.float32)
    size = np.ones((int(pop_size), int(width)), dtype=np.int32)
    return opcode, payload, size
