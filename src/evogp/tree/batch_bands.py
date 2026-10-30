"""The batch route's band policy: how many merge bands a device can actually LOAD.

A banded merge (``evoxc.batch_functions(..., bands=K)``) emits ONE exported
dispatch per band, and every IREE HAL driver caps the buffer bindings a single
exported dispatch may use, so a band count that is fine on one backend is
unloadable on another:

* CUDA and ROCm/HIP bind at most 16 tensors per exported dispatch
  (``iree/runtime/src/iree/hal/drivers/{cuda,hip}/native_executable.c``), so the
  measured safe count is ``K <= 15`` and a band must carry at least 8 functions;
* the local-task CPU driver allows 64 bindings, and ``K <= 31`` is measured safe;
* the evoxc reference interpreter (``numpy``) and the Vulkan HAL are UNCAPPED.

The practical consequence -- why this module exists -- is that the band-SIZE model
(``EVOGP_BATCH_BAND_SIZE=16``, ``K = ceil(n / 16)``) reaches ``K = 63`` at a pop-1000
population, which provably cannot load on a GPU HAL.  So this module answers two
questions for :mod:`evogp.tree.batch_programs`: what to ASK evoxc for
(:func:`plan_band_spec` / :func:`resolve_band_plan` -- flat, an explicit ``K``, the
size model or ``"auto"``, with a modelled count clamped into what the device can
load) and what an ``auto`` resolution actually produced
(:func:`clamp_auto_band_count` on ``merged.batch_report.bands``, applied by
:func:`merge_with_spec` with the one-remerge remedy).

An EXPLICIT band count is never silently clamped: a ``K`` the device cannot load
raises ``ValueError`` naming the variable, the real limit, the runtime source and
the remedy, because a plausible band count that cannot load is a configuration
bug, not a preference.

Nothing here imports ``evoxc``, torch or a graph helper: the tables and the
arithmetic are host-side policy, so every branch is testable on a CPU box and
:func:`merge_with_spec` takes the caller's merge callable.  The measured envelope
behind the tables lives in ``BATCH_EVAL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from . import batch_knobs
from .batch_knobs import BANDS_ENV
from .evoxc_engine import _gpu_device_kind

#: Buffer bindings ONE exported dispatch may use, per IREE HAL device kind (the
#: ``{hip,cuda}`` drivers bind at most 16, ``local_task`` allows 64); ``None`` = uncapped.
DISPATCH_BINDING_LIMITS: Dict[str, Optional[int]] = {
    "cuda": 16,
    "rocm": 16,
    "cpu": 64,
    "vulkan": None,
}

#: Measured SAFE band count per kind: one below the GPU binding limit, 31 on the
#: CPU HAL (the per-dispatch I/O of a band overflows at the limit itself).
SAFE_BAND_COUNTS: Dict[str, Optional[int]] = {
    "cuda": 15,
    "rocm": 15,
    "cpu": 31,
    "vulkan": None,
}

#: Smallest band worth its own dispatch: a GPU dispatch pays a launch plus its
#: binding setup, so sub-8-function GPU bands lose more than they save.
MIN_BAND_SIZE: Dict[str, int] = {"cuda": 8, "rocm": 8, "cpu": 1, "vulkan": 1}
#: The device kinds whose compile target is a GPU HAL, i.e. the kinds whose device
#: code an IREE GPU-only codegen lever applies to.  It is the same set the binding
#: tables above know (an evoxc ``Device.kind``), so the route reads it HERE rather
#: than spelling "rocm"/"cuda" again wherever a target-dependent policy is decided.
GPU_DEVICE_KINDS: Tuple[str, ...] = ("cuda", "rocm")

#: ``kind -> (how a message names the HAL, where its binding limit is enforced)``.
_RUNTIME: Dict[str, Tuple[str, str]] = {
    "cuda": ("CUDA", "iree runtime src/iree/hal/drivers/cuda/native_executable.c"),
    "rocm": ("ROCm/HIP", "iree runtime src/iree/hal/drivers/hip/native_executable.c"),
    "cpu": (
        "local-task CPU",
        "iree runtime src/iree/hal/drivers/local_task/native_executable.c",
    ),
}

#: The band request spellings a :class:`BandSpec` can carry.
BAND_REQUESTS = ("flat", "explicit", "size", "auto")


def _hal(kind: str) -> str:
    """The HAL driver's name for ``kind`` (the upper-cased spelling when unknown)."""
    return _RUNTIME.get(kind, (kind.upper() or "unknown", ""))[0]


def _source(kind: str) -> str:
    """``" (<runtime source>)"`` for a known kind, else ``""``."""
    origin = _RUNTIME.get(kind, ("", ""))[1]
    return f" ({origin})" if origin else ""


def _binding_phrase(kind: str) -> str:
    """``one exported dispatch may bind at most N tensors`` for a capped kind."""
    limit = int(DISPATCH_BINDING_LIMITS.get(kind) or 0)
    return f"one exported dispatch may bind at most {limit} tensors"


@dataclass(frozen=True)
class BandPolicy:
    """What one (backend, device kind) pair allows a banded merge to do.

    Attributes:
        backend: the evoxc backend name (``"numpy"`` / ``"iree"``); numpy and an
            empty name are always uncapped.
        device_kind: the codegen device kind (``"rocm"`` / ``"cuda"`` / ``"cpu"`` /
            ``"vulkan"``), i.e. which IREE HAL driver binds the dispatch.
        max_bands: the safe band count, or ``None`` when the device has no
            dispatch-binding cap (the interpreter, Vulkan, an unknown kind).
        min_band_size: the smallest band worth its own dispatch on this device.
    """

    backend: str
    device_kind: str
    max_bands: Optional[int]
    min_band_size: int

    @property
    def uncapped(self) -> bool:
        """``True`` when this device imposes no band-count cap."""
        return self.max_bands is None

    @property
    def label(self) -> str:
        """``"<backend>/<device kind>"`` -- how every note and refusal names it."""
        return f"{self.backend}/{self.device_kind}"

    @property
    def reason(self) -> str:
        """The binding cap in one sentence, naming the runtime source."""
        kind = self.device_kind
        if self.max_bands is None:
            if self.backend == "numpy":
                return (
                    "the evoxc reference interpreter has no buffer-binding cap, so "
                    "the band count is only bounded by the function count"
                )
            return (
                f"no exported-dispatch binding limit is known for {_hal(kind)} on "
                "this backend, so the band count is only bounded by the function count"
            )
        text = (
            "a banded merge emits one exported dispatch per band and "
            f"{_binding_phrase(kind)} on the {_hal(kind)} HAL{_source(kind)}, so the "
            f"band count must satisfy K <= {int(self.max_bands)} here"
        )
        if int(self.min_band_size) > 1:
            text += (
                f"; a band also has to carry at least {int(self.min_band_size)} "
                "merged functions, so a smaller population supports one band only"
            )
        return text

    def cap(self, n_functions: int) -> int:
        """The largest band count this device can load for ``n_functions``.

        ``n_functions`` when there is no binding cap (the caller still clamps into
        ``1 <= K <= n``), else ``max(1, min(max_bands, n, max(1, n // min_band_size)))``
        -- the band limit, the function count and the per-band size floor, whichever
        bites first.
        """
        total = int(n_functions)
        if self.max_bands is None:
            return total
        floor = max(1, total // max(1, int(self.min_band_size)))
        return max(1, min(int(self.max_bands), total, floor))

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-ish snapshot of the policy (it travels in a plan's report)."""
        return {
            "backend": self.backend,
            "device_kind": self.device_kind,
            "max_bands": self.max_bands,
            "min_band_size": int(self.min_band_size),
            "uncapped": self.uncapped,
        }


def resolve_band_policy(
    backend: Optional[str] = None, device: Optional[Any] = None
) -> BandPolicy:
    """The :class:`BandPolicy` of one backend/device pair (no compile, no I/O).

    ``"numpy"`` (or ``None`` / empty) is UNCAPPED and takes its ``device_kind`` from
    ``device.kind`` when given, else ``"cpu"``.  ``"iree"`` takes the kind from
    ``device.kind`` when given, else from the installed torch build
    (:func:`evogp.tree.evoxc_engine._gpu_device_kind`), and applies the tables above.
    An unknown backend name or device kind is uncapped: the route resolves both
    before any merge, so that is "no limit known", never a guess.
    """
    name = "" if backend is None else str(backend).strip().lower()
    kind = getattr(device, "kind", None) if device is not None else None
    kind = str(kind).strip().lower() if kind else ""
    if name != "iree":
        return BandPolicy(name or "numpy", kind or "cpu", None, 1)
    kind = kind or str(_gpu_device_kind()).strip().lower()
    if kind not in DISPATCH_BINDING_LIMITS:
        return BandPolicy(name, kind, None, 1)
    return BandPolicy(
        name, kind, SAFE_BAND_COUNTS.get(kind), int(MIN_BAND_SIZE.get(kind, 1))
    )


@dataclass(frozen=True)
class BandSpec:
    """What ONE merge asks evoxc for, plus the policy that vetted the answer.

    Attributes:
        requested: the request spelling, one of :data:`BAND_REQUESTS` -- ``"flat"``
            (a single merge), ``"explicit"`` (a count the caller /
            ``EVOGP_BATCH_BANDS`` pinned), ``"size"`` (the band-SIZE model decided)
            or ``"auto"`` (evoxc's cost model decides).
        num_bands: the resolved count for ``"flat"`` / ``"explicit"`` / ``"size"``
            (``>= 1``), already clamped by ``policy``; ``None`` for ``"auto"``,
            whose count is only known after the merge.
        clamped: whether ``policy`` lowered a modelled/auto band count.
        note: the human explanation of that clamp, else ``None``.
        policy: the :class:`BandPolicy` the spec was vetted against.
    """

    requested: str
    num_bands: Optional[int]
    clamped: bool
    note: Optional[str]
    policy: BandPolicy

    @property
    def evoxc_bands(self) -> Any:
        """The ``bands=`` argument to hand to ``evoxc.batch_functions``.

        ``"auto"`` for an auto request, an ``int`` for a real band split, and
        ``None`` for the flat merge -- including a resolved ``K == 1``, which is the
        flat emitter and must travel as ``None`` to stay byte-identical to it.
        """
        if self.requested == "auto":
            return "auto"
        if self.num_bands is None or int(self.num_bands) <= 1:
            return None
        return int(self.num_bands)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-ish snapshot of the spec (it travels in a plan's report)."""
        return {
            "requested": self.requested,
            "num_bands": None if self.num_bands is None else int(self.num_bands),
            "clamped": bool(self.clamped),
            "note": self.note,
        }


def _clamp_note(
    chosen: int, capped: int, policy: BandPolicy, *, what: str, remedy: str
) -> str:
    """The explanation of a clamp, naming ``what`` chose ``chosen`` bands."""
    kind = policy.device_kind
    if policy.max_bands is not None and int(capped) == int(policy.max_bands):
        detail = (
            "a banded merge emits one exported dispatch per band and "
            f"{_binding_phrase(kind)} on the {_hal(kind)} HAL{_source(kind)}, so at "
            f"most {int(capped)} bands can load here"
        )
    else:
        detail = (
            f"a band has to carry at least {int(policy.min_band_size)} merged "
            f"functions on this backend, so at most {int(capped)} bands can load here"
        )
    return (
        f"{what} chose {int(chosen)} bands but {policy.label} can only load "
        f"{int(capped)}: {detail}; the merge was clamped to {int(capped)} bands. "
        f"{remedy}"
    )


def _reject_message(requested: int, total: int, policy: BandPolicy) -> str:
    """The EXPLICIT-``K`` refusal: what cannot load, why, and what to do."""
    capped = policy.cap(total)
    kind = policy.device_kind
    clause = (
        "a banded merge emits one exported dispatch per band and "
        f"{_binding_phrase(kind)} on the {_hal(kind)} HAL{_source(kind)}"
    )
    other = (
        "the local-task CPU backend's limit is K <= 31 and numpy has no binding cap"
        if kind != "cpu"
        else "the CUDA/ROCm HAL's limit is K <= 15 and numpy has no binding cap"
    )
    if policy.max_bands is not None and int(capped) == int(policy.max_bands):
        clause += f", so the band count must satisfy K <= {int(capped)} here"
        remedy = (
            f"Lower it ({BANDS_ENV}={int(capped)} or less) or raise "
            f"{batch_knobs.BAND_SIZE_ENV} to {int(policy.min_band_size)} or more so "
            "fewer bands are chosen"
        )
    else:
        clause += (
            ", and a band has to carry at least "
            f"{int(policy.min_band_size)} merged functions, so {int(total)} functions "
            f"support at most K <= {int(capped)} bands here"
        )
        remedy = (
            f"Lower it ({BANDS_ENV}={int(capped)} or less), or merge more functions "
            "(a larger population, or fewer deduped programs)"
        )
    return (
        f"{BANDS_ENV}={int(requested)} cannot load on the {policy.label} backend: "
        f"{clause}. {remedy}; {other}."
    )


def plan_band_spec(
    n_functions: int,
    *,
    bands: Optional[Any] = None,
    band_size: Optional[int] = None,
    switch_ops: Optional[bool] = None,
    policy: Optional[BandPolicy] = None,
) -> BandSpec:
    """Resolve ONE merge's :class:`BandSpec` for ``n_functions`` merged functions.

    Precedence (the batch route's documented chain, now policy-aware):

    1. ``switch_ops`` on -> the flat merge (banding is never combined with it);
    2. ``bands="auto"`` -> ``requested="auto"``, the evoxc cost model decides;
    3. an explicit ``K`` / flat spelling -> the flat merge for ``K == 1``, else
       ``"explicit"`` with ``K`` validated into ``1..n_functions`` (the existing
       ``ValueError`` naming the variable) and then against
       ``policy.cap(n_functions)`` -- an unloadable ``K`` RAISES, never clamps;
    4. otherwise the band-SIZE model ``K = ceil(n_functions / band_size)`` clamped
       into ``1..n_functions``, then lowered to ``policy.cap`` with ``clamped=True``
       and a note when the device cannot load it.

    ``policy=None`` means UNCAPPED (:func:`resolve_band_policy` with no backend).
    """
    total = int(n_functions)
    if total < 1:
        raise ValueError(f"n_functions={total} must be >= 1")
    policy = resolve_band_policy() if policy is None else policy
    if batch_knobs.resolve_switch_ops(switch_ops):
        return BandSpec("flat", 1, False, None, policy)
    explicit = batch_knobs.resolve_bands(bands)
    if isinstance(explicit, str):  # the ``auto`` spelling, the only string answer
        return BandSpec("auto", None, False, None, policy)
    if explicit is not None:
        count = int(explicit)
        if not 1 <= count <= total:
            raise ValueError(
                f"{BANDS_ENV}={count} is outside 1..{total} (the number of merged "
                "functions)"
            )
        if count == 1:
            return BandSpec("flat", 1, False, None, policy)
        if count > policy.cap(total):
            raise ValueError(_reject_message(count, total, policy))
        return BandSpec("explicit", count, False, None, policy)
    size = batch_knobs.resolve_band_size(band_size)
    modelled = max(1, min(total, -(-total // size)))
    capped = policy.cap(total)
    if modelled <= capped:
        return BandSpec("size", modelled, False, None, policy)
    note = _clamp_note(
        modelled,
        capped,
        policy,
        what=f"the band-size model ({batch_knobs.BAND_SIZE_ENV}={size})",
        remedy=(
            f"Raise {batch_knobs.BAND_SIZE_ENV} so fewer bands are chosen, or pass an "
            f"explicit {BANDS_ENV}=K with K <= {int(capped)} to pick the partition "
            "yourself."
        ),
    )
    return BandSpec("size", capped, True, note, policy)


def resolve_band_plan(
    n_functions: int,
    *,
    bands: Optional[Any] = None,
    band_size: Optional[int] = None,
    switch_ops: Optional[bool] = None,
    backend: Optional[str] = None,
    device: Optional[Any] = None,
) -> BandSpec:
    """The call :mod:`evogp.tree.batch_programs` makes: build the policy, then the spec."""
    return plan_band_spec(
        n_functions,
        bands=bands,
        band_size=band_size,
        switch_ops=switch_ops,
        policy=resolve_band_policy(backend, device),
    )


def clamp_auto_band_count(
    chosen: int, n_functions: int, policy: BandPolicy
) -> Tuple[int, Optional[str]]:
    """Clamp an ``auto`` resolution (``merged.batch_report.bands``) to the policy.

    The cost model is CPU-calibrated and knows nothing about dispatch bindings, so
    it can return an unloadable count.  The answer is ``(min(chosen,
    policy.cap(n_functions)), note_or_None)``, and the note names the auto choice,
    the clamp, the binding cap, the backend kind and the remedy.
    """
    capped = policy.cap(int(n_functions))
    picked = max(1, int(chosen))
    if picked <= capped:
        return picked, None
    note = _clamp_note(
        picked,
        capped,
        policy,
        what="auto (the evoxc cost model)",
        remedy=(
            f"Pass an explicit {BANDS_ENV}=K with K <= {int(capped)} to pin the "
            f"partition (or raise {batch_knobs.BAND_SIZE_ENV} and drop the 'auto' "
            "spelling, so the band-size model asks for fewer bands)."
        ),
    )
    return capped, note


def chunk_band_count(
    global_k: int, chunk_n: int, n_functions: int, policy: BandPolicy
) -> int:
    """The proportional split of a global band count over one chunk.

    ``max(1, min(chunk_n, policy.cap(chunk_n), ceil(global_k * chunk_n /
    n_functions)))``: a chunk keeps its share of the global bands but never more
    bands than it has functions or than the device can load.  Row order is
    unaffected (chunks are contiguous slices of the global function order), so only
    the dispatch count moves.
    """
    total = int(n_functions)
    n = int(chunk_n)
    if total < 1 or n < 1:
        raise ValueError(
            "chunk_band_count needs n_functions >= 1 and chunk_n >= 1, got "
            f"n_functions={total}, chunk_n={n}"
        )
    proportional = max(1, -(-(int(global_k) * n) // total))
    return max(1, min(n, policy.cap(n), proportional))


def chunk_band_spec(
    spec: BandSpec,
    chunk_n: int,
    n_functions: int,
    *,
    band_size: Optional[int] = None,
    policy: Optional[BandPolicy] = None,
) -> BandSpec:
    """The :class:`BandSpec` of ONE chunk of a chunked plan.

    A chunked plan repeats the plan's band request on every chunk (what
    ``EVOGP_BATCH_CHUNK_BANDS`` opts into), so the spelling travels unchanged while
    the count is resized: ``"auto"`` stays ``"auto"`` (clamped per chunk after the
    merge), ``"flat"`` stays flat, an explicit ``K`` becomes
    :func:`chunk_band_count`, and the band-SIZE model is re-applied to the chunk
    (``band_size=None`` re-reads the knob).
    """
    total = int(n_functions)
    n = int(chunk_n)
    if n < 1:
        raise ValueError(f"chunk_n={n} must be >= 1")
    policy = spec.policy if policy is None else policy
    if spec.requested == "auto":
        return BandSpec("auto", None, False, None, policy)
    if spec.requested == "flat":
        return BandSpec("flat", 1, False, None, policy)
    if spec.requested == "explicit":
        global_k = int(spec.num_bands or 1)
        proportional = max(1, -(-(global_k * n) // max(1, total)))
        count = chunk_band_count(global_k, n, total, policy)
        if count == proportional:
            return BandSpec("explicit", count, False, None, policy)
        note = _clamp_note(
            proportional,
            count,
            policy,
            what=f"the proportional split of {BANDS_ENV}={global_k}",
            remedy=(
                "Pass a smaller global band count, or merge the functions in one body "
                f"(unset {batch_knobs.CHUNK_BANDS_ENV}) to keep the global split."
            ),
        )
        return BandSpec("explicit", count, True, note, policy)
    size = batch_knobs.resolve_band_size(band_size)
    modelled = max(1, min(n, -(-n // size)))
    capped = policy.cap(n)
    if modelled <= capped:
        return BandSpec("size", modelled, False, None, policy)
    note = _clamp_note(
        modelled,
        capped,
        policy,
        what=f"the band-size model ({batch_knobs.BAND_SIZE_ENV}={size}) on this chunk",
        remedy=(
            f"Raise {batch_knobs.BAND_SIZE_ENV} so fewer bands are chosen, or pass an "
            f"explicit {BANDS_ENV}=K with K <= {int(capped)}."
        ),
    )
    return BandSpec("size", capped, True, note, policy)


def merge_with_spec(
    merge: Callable[..., Any],
    graphs: Sequence[Any],
    axes: Any,
    spec: BandSpec,
    *,
    switch_ops: bool,
    alignment: str,
    auto_total: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Merge ``graphs`` under ``spec`` and stamp the band-policy report keys.

    ``merge`` is the caller's ``evoxc.batch_functions`` wrapper (this module never
    imports evoxc), called as ``merge(graphs, axes, switch_ops=..., bands=...,
    alignment=...)``.

    An ``"auto"`` spec cannot know its band count before the merge, so its
    resolution is clamped by :func:`clamp_auto_band_count` and, when the device
    cannot load it, the merge is redone ONCE with the clamped count -- both merges
    stay inside the caller's ``merge`` stage.  The AUTO merge's ``cost_estimate`` /
    ``flat_cost_estimate`` say what the model thought about the CHOICE, so they
    survive onto the final report even when the clamped re-merge reports no model
    run.

    The returned report is ``batch_report.as_dict()`` plus exactly the five policy
    keys ``band_requested`` / ``band_requested_bands`` / ``band_clamped`` /
    ``band_note`` / ``band_policy``.
    """
    merged = merge(
        graphs, axes, switch_ops=switch_ops, bands=spec.evoxc_bands, alignment=alignment
    )
    report = merged.batch_report.as_dict()
    requested_bands: Optional[int] = None
    clamped = bool(spec.clamped)
    note = spec.note
    if spec.requested == "auto":
        requested_bands = int(report.get("bands") or 1)
        capped, auto_note = clamp_auto_band_count(
            requested_bands, int(auto_total), spec.policy
        )
        if auto_note is not None:
            cost_estimate = report.get("cost_estimate")
            flat_cost_estimate = report.get("flat_cost_estimate")
            merged = merge(
                graphs,
                axes,
                switch_ops=switch_ops,
                bands=None if capped <= 1 else capped,
                alignment=alignment,
            )
            report = merged.batch_report.as_dict()
            report["cost_estimate"] = cost_estimate
            report["flat_cost_estimate"] = flat_cost_estimate
            clamped = True
            note = auto_note
    report["band_requested"] = spec.requested
    report["band_requested_bands"] = requested_bands
    report["band_clamped"] = bool(clamped)
    report["band_note"] = note
    report["band_policy"] = spec.policy.as_dict()
    return merged, report


def aggregate_band_report(
    spec: BandSpec, reports: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """The plan-level band keys of a CHUNKED plan (from its per-chunk reports).

    The request spelling is the plan's, ``band_requested_bands`` the SUM of the
    per-chunk auto counts before clamping (``None`` when no chunk ran the model),
    ``band_clamped`` true when any chunk was clamped, and ``band_note`` joins the
    clamped chunks' notes with ``" | "``.
    """
    notes = [str(report["band_note"]) for report in reports if report.get("band_note")]
    requested = [
        int(report["band_requested_bands"])
        for report in reports
        if report.get("band_requested_bands") is not None
    ]
    return {
        "band_requested": spec.requested,
        "band_requested_bands": sum(requested) if requested else None,
        "band_clamped": any(bool(report.get("band_clamped")) for report in reports),
        "band_note": " | ".join(notes) if notes else None,
        "band_policy": spec.policy.as_dict(),
    }
