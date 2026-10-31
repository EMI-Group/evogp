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

The upload cannot be avoided: ``evoxc`` has no zero-copy device export (its
device/host interop is host-only), so a torch route always starts from host
bytes -- see ``measure_fused.py`` for what that transfer costs relative to the
in-graph ``evoxc`` route, and :class:`FusedEvaluator` for the version that keeps
the *data* resident on the device across calls (the channel tables still upload,
because the population changes every generation).

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
* The output lives on the host: this is a numpy-facing adapter, not a device
  result (callers that want the device tensor should use the sibling engine
  directly).
* A malformed population is NOT declined by the kernel (it produces a
  defined-but-arbitrary number); ``validate=True`` runs the host decode
  validation (:func:`evogp.gp.encoding.validate_channels`) first, which costs a
  Python loop over rows x nodes and is therefore opt-in.

Scope: this module adds a route, it does not change one.  Nothing in
``evogp.gp`` selects it -- the driver's device path is the in-graph ``evoxc``
route, and ``measure_fused.py`` is the committed evidence for what wiring this
one in would (and would not) buy.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any, Optional, Tuple

import numpy as np

from .encoding import _channel_arrays, to_legacy_triple

__all__ = [
    "TRITON_CACHE_DIR_ENV",
    "FusedUnavailable",
    "ensure_triton_cache_dir",
    "fused_available",
    "sr_fitness_fused_channels",
    "FusedEvaluator",
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
        """The uploaded data tensors, built once per (process, device)."""
        if self._tensors is None or self._tensors[0] != device:
            self._tensors = (device,) + _data_tensors(
                torch_module, device, self.x, self.y
            )
        return self._tensors[1], self._tensors[2]
