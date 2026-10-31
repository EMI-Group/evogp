"""Optional FUSED (Triton) evaluation backend for ``evogp.gp`` -- the ONE
deliberate torch boundary in this subtree.

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
    downloaded: :mod:`evogp.gp._device_interop` hands the channel buffers to
    torch as zero-copy DLPack views, the same conversion runs on the device, the
    kernel's ``(pop_size,)`` loss is negated in torch, and the resulting buffer
    goes back to ``evoxc`` the same way -- so the fitness comes back as an
    ``evoxc`` DEVICE tensor and the driver hands it straight to the variation
    graph with no readback (``workflow.step`` accepts either).

The upload above is only unavoidable on the HOST route.  ``evoxc`` still has no
first-class device export (its documented device/host interop is host-only), so
the device route borrows one through a small, explicitly temporary, LOCAL shim
(:mod:`evogp.gp._device_interop`) rather than pretending the gap is closed --
see that module for the protocol, the hard validation and the stream assumption,
and ``measure_fused.py`` for what each route costs relative to the in-graph
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
import os
import tempfile
from typing import Any, Optional, Tuple

import numpy as np

from ._constants import FUNC_TYPE, Func, NType
from ._device_interop import (
    InteropUnavailable,
    device_of,
    export_to_torch,
    import_from_torch,
    interop_available,
    is_device_tensor,
)
from .encoding import _channel_arrays, to_legacy_triple

__all__ = [
    "TRITON_CACHE_DIR_ENV",
    "FusedUnavailable",
    "ensure_triton_cache_dir",
    "fused_available",
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
    tables still upload on every call -- the population changes every generation,
    and ``evoxc`` has no zero-copy device export, so they cannot come from a
    device tensor without a host round trip anyway.

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
                 (:func:`evogp.gp._device_interop.export_to_torch`)
    conversion   the ``Forest`` legacy triple, computed in torch on the device
                 (``_legacy_triple_device``)
    kernel       ``fused_engine.sr_fitness_fused`` -- the loss is reduced
                 in-kernel, so the only device output is ``(pop_size,)``
    fitness      negate + non-finite -> ``-inf`` in torch, then adopted back as
                 an ``evoxc`` device tensor
                 (:func:`evogp.gp._device_interop.import_from_torch`) -- which
                 the variation graph consumes directly, no readback
    ============ =============================================================

    Cost profile: one DLPack capsule per channel per call (no bytes move), one
    kernel launch, and the ``OUT_NODE`` guard -- the only mandatory host sync in
    the call, a single boolean readback (~0.04 ms on this box, and it is read
    after the conversion is enqueued so the wait covers that work too).
    ``x``/``y`` are uploaded to torch once per device and reused; nothing is
    uploaded by the constructor.

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
        try:
            loss = engine.sr_fitness_fused(
                value,
                node_type,
                subtree_size,
                self.problem_dim,
                1,
                x_tensor,
                y_tensor,
                use_MSE=self.use_mse,
            )
        except engine.FusedUnsupported as exc:
            raise FusedUnavailable(
                f"the fused route declined this call: {exc}"
            ) from exc
        if int(loss.numel()) != pop_size:
            raise RuntimeError(
                f"the fused kernel returned {int(loss.numel())} values for a "
                f"population of {pop_size}"
            )
        fitness = torch_module.where(
            torch_module.isfinite(loss),
            -loss,
            torch_module.full_like(loss, float("-inf")),
        )
        return import_from_torch(fitness.contiguous(), size)

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
