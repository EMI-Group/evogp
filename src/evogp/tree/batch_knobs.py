"""The batch route's knob vocabulary, variant rules and decline type.

One of the four modules the opt-in batch evaluation route lives in:

* **this module** -- the ``EVOGP_BATCH_*`` constants, the ``resolve_*`` helpers
  that read them per call, the ``value`` / ``loss`` variant spellings with
  :func:`_normalize_variant`, and :class:`BatchUnsupported`.
* :mod:`evogp.tree.batch_programs` -- host-side program handling: the input /
  label conversions, the per-function graph bodies, the structural dedup, the
  trace and the merge, the plan digest, plus the :class:`BatchPlan` /
  :class:`BatchChunk` / :class:`BatchExecutable` records.
* :mod:`evogp.tree.batch_chunks` -- the compile-side chunk orchestration (the
  partition arithmetic and the ``lower`` / ``compile`` / ``load`` pool).
* :mod:`evogp.tree.batch_engine` -- the PUBLIC facade (import that one, not the
  two siblings): the three front tables, :func:`plan_batch` / :func:`build_batch` /
  :func:`evaluate_batch` / :func:`sr_fitness_batch` / :func:`describe_plan`,
  :func:`batch_cache_stats` / :func:`batch_content_stats` and the stage reporting,
  which re-exports everything defined here.
* :mod:`evogp.tree.batch_stages` -- the per-stage timer surface the facade's
  ``batch_stage_stats()`` reports from.
* :mod:`evogp.tree.batch_bands` -- the band/binding policy: which band counts a
  backend and device kind can actually LOAD (the IREE HAL dispatch-binding cap),
  the ``auto`` clamp and the per-chunk band split.  It imports this module, never
  the other way round.

This module imports nothing but :func:`evogp.tree.evoxc_engine._env_int` (no
``evoxc``, no torch, no graph work), so it stays importable on its own and both
siblings may import it without a cycle.

Every knob is read per call unless its comment says otherwise, and its precedence
is ``explicit argument > environment > default``; an unusable value raises
``ValueError`` naming the variable.  The scheme, the measured economics and the
knob table live in ``BATCH_EVAL.md``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from .evoxc_engine import _env_int

# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
#: The ``EVOGP_EVAL_MODE`` spelling of this route.  Deliberately NOT consulted by
#: this module (see the module docstring): the wiring task adds it to
#: :data:`evogp.tree.dense_engine.EVAL_MODES` and calls the two entry points here.
EVAL_MODE = "batch"
#: Band SIZE of the merge (``EVOGP_BATCH_BAND_SIZE``): ``K = ceil(n / size)``
#: bands, clamped to ``1 <= K <= n``.  16 is the measured minimax default: over the
#: nine warm columns of the band sweep it is the only setting never more than ~6 %
#: off any column (pooled geomean 1.0185, worst column 1.061 -- against 1.0223 /
#: 1.137 for 32 and 1.529 for the flat merge), and at the 225-245 deduped function
#: count of a pop-256 population it lands on ``K = 15-16``, i.e. right at the
#: ``K <= 15`` dispatch-binding cap, whose 8-functions-per-band floor also makes a
#: band size below 8 pointless on a GPU HAL.  Fewer/larger bands lose on the
#: per-row ``forward`` path (up to 1.5x against the flat merge), which is the
#: workload that decides the default; the wide-row loss path is flat in ``K``.
#: Pinning the band COUNT at the device binding cap is not a new partition where
#: that cap already bites: at a pop-256 population the size model is clamped to 15
#: anyway, so ``EVOGP_BATCH_BANDS=15`` produces bit-identical ``band_sizes``
#: (``[15]x15`` at 225 functions, ``[16]x7 + [15]x8`` at 232, ``[17]x5 + [16]x10``
#: on the per-row plan at 245) and identical ``emitted_ops`` (5 037 / 4 153 / 4 540
#: and 5 475 / 4 568 / 5 861).  At ``p64`` the cap is ``K = 7`` (15 would be
#: illegal: an explicit ``K`` needs ``pop >= 8K``), where the pin DOES change the
#: plan (``[16,16,16,15]`` with 1 393 ops -> ``[9]x7`` with 1 463 ops) yet measures
#: +0.04 % / +1.3 % / -4.6 % on ``sr_fitness`` / ``batch_forward`` / ``forward``,
#: inside this box's ~5 % cross-config noise floor.  No pin beat the shipped size
#: anywhere, and the oracle deviation is exactly identical in all four configs of
#: all nine cell/workload rows -- the partition is a scheduling decision, never a
#: numeric one.
BAND_SIZE_ENV = "EVOGP_BATCH_BAND_SIZE"
DEFAULT_BAND_SIZE = 16
#: Explicit band count (``EVOGP_BATCH_BANDS``): an int ``K`` with
#: ``1 <= K <= n_functions``, the ``auto`` spelling (let evoxc's static cost model
#: pick the partition), or one of the flat spellings ``none`` / ``flat`` / ``0``,
#: which mean an explicit single band (``K = 1``, the flat merge).
#: Unset/empty means "use the band-size model above".  An explicit ``K`` or an
#: ``auto`` resolution that the device cannot LOAD is refused/clamped by
#: :mod:`evogp.tree.batch_bands`; :func:`resolve_bands` itself stays
#: backend-agnostic.
BANDS_ENV = "EVOGP_BATCH_BANDS"
#: The ``auto`` spelling of ``EVOGP_BATCH_BANDS``: evoxc groups the functions by
#: structural fingerprint and scores the candidate partitions with its static
#: work-vs-dispatch cost model, reporting the choice on
#: ``merged.batch_report`` (``band_strategy == "auto"``, ``cost_estimate`` /
#: ``flat_cost_estimate``).  The model is CPU-calibrated and knows nothing about a
#: device's dispatch-binding cap, so the resolved count is clamped by
#: :func:`evogp.tree.batch_bands.clamp_auto_band_count` before it is used.
BANDS_AUTO = "auto"
#: Banded compile chunks (``EVOGP_BATCH_CHUNK_BANDS``, off by default): when on, a
#: band request and the chunked compile path are COMPOSED -- each chunk repeats the
#: plan's band request (``auto`` per chunk, the proportional share of an explicit
#: ``K``, the size model re-applied to the chunk) instead of the band request
#: forcing the single-body path.  Off keeps the documented meaning of both knobs.
CHUNK_BANDS_ENV = "EVOGP_BATCH_CHUNK_BANDS"
DEFAULT_CHUNK_BANDS = False
#: Op-switching (``EVOGP_BATCH_SWITCH_OPS``): off by default -- on the measured GP
#: workload it shares nothing (``report["switched"]`` empty) and changes no op
#: count, so it stays a knob; when on it forces the flat merge.
SWITCH_OPS_ENV = "EVOGP_BATCH_SWITCH_OPS"
#: Class alignment of a ``switch_ops`` merge (``EVOGP_BATCH_ALIGNMENT``); a pure
#: performance heuristic inside evoxc -- it can never change the numbers.
ALIGNMENT_ENV = "EVOGP_BATCH_ALIGNMENT"
DEFAULT_ALIGNMENT = "variant"
ALIGNMENTS = ("variant", "occurrence", "auto")
#: Population threshold (``EVOGP_BATCH_MIN_POP``): below it the merge (a compile
#: is linear in the function count, and the whole route's win is amortizing ONE
#: invocation) is not worth attempting, so the route declines.
#: The default is the measured BREAK-EVEN, not a safety floor: at ``pop=4``
#: batch+iree is 659 ms cold against 48 ms for specialized+numpy and 0.76 ms
#: warm against 0.63 ms (1.2x SLOWER, because a merged compile cannot amortize
#: over four functions), while from ``pop=64`` up it is 1.10 ms warm against
#: 8.42 ms (7.6x faster) for a 3.30 s compile.  Raise ``EVOGP_BATCH_MIN_POP``
#: (or pass ``value=`` to :func:`resolve_min_pop`) to admit smaller populations
#: on purpose.
MIN_POP_ENV = "EVOGP_BATCH_MIN_POP"
DEFAULT_MIN_POP = 64
#: Entries of the two front tables (``EVOGP_BATCH_CACHE_SIZE``, read at IMPORT
#: time like ``EVOGP_EVOXC_CACHE_SIZE``).  Small on purpose: a plan entry holds
#: the traced per-function graphs, the merged graph AND the compiled executable,
#: and a training loop presents a new population every generation.
CACHE_SIZE_ENV = "EVOGP_BATCH_CACHE_SIZE"
DEFAULT_CACHE_SIZE = 8
#: Compile workers (``EVOGP_BATCH_COMPILE_WORKERS``): how many contiguous chunks the
#: deduped function list is split into, whose ``lower`` / ``compile`` / ``load``
#: bodies are then staged in a thread pool.  ``1`` (and ``0``, its alias) is the
#: original single-executable plan verbatim -- one merged graph, one executable.
#: 4 is deliberately modest: each live chunk holds its own device executable, and
#: an iree session leaks device buffers per live executable (see ``BATCH_EVAL.md``
#: "Memory, threads and iree fragility"), so the knob is a compile-wall lever for
#: large populations, not something to raise by default.
#: The measured trade (``p256-L32`` / ``p256-L64`` / ``p256-L128-d7``): the pool
#: makes the COLD call 2.64x / 2.94x / 4.58x cheaper (10 725.86 / 11 957.38 /
#: 38 513.95 ms at ``1`` against 4 064.48 / 4 071.43 / 8 410.96 ms at ``4``) and
#: the WARM ``sr_fitness`` call 1.87x / 1.69x / 1.46x dearer (1.9941 / 2.2267 /
#: 3.4851 ms against 3.7250 / 3.7720 / 5.0947 ms), because a chunked call does four
#: upload / run / read-back trios instead of one.  Break-even -- the cold
#: difference repaid in warm calls of ONE population -- is 3 849 / 5 103 / 18 702
#: (3 981 / 5 083 / 20 648 from the mins), which a GP loop never reaches: it
#: presents a new structure multiset every generation (7 of 8 generations
#: recompile, 8 distinct signatures in 8 steps), while one short run of a big
#: population is aggregate-cold-sensitive and wants the shipped 4.  Where the pool
#: costs the winning niche nothing it is inert: below ``MIN_CHUNK_FUNCTIONS``
#: (32 functions) it is one chunk, and the ``p64`` fixed-population cell (63
#: functions) plans ``chunk_count 1`` / ``chunk_sizes [63]`` with no trade at all
#: (cold 2 268.96 -> 2 274.46 ms, warm 1.1494 -> 1.1141 ms).
#: ``EVOGP_BATCH_COMPILE_WORKERS=1`` is the documented escape hatch for a workload
#: that evaluates one fixed structure multiset more often than those break-evens
#: (~3.8k / ~5.1k / ~18.7k warm calls).
COMPILE_WORKERS_ENV = "EVOGP_BATCH_COMPILE_WORKERS"
DEFAULT_COMPILE_WORKERS = 4
#: Dispatch formation of a merged graph's compile
#: (``EVOGP_BATCH_DISPATCH_FORMATION``): ``auto`` (default), ``aggressive`` or
#: ``default`` (alias ``off``).
#: It maps onto evoxc's iree compile option of the same name, whose only spelling
#: that changes anything is ``aggressive`` -- it derives
#: ``--iree-dispatch-creation-enable-aggressive-fusion`` (596 -> 548 IREE
#: dispatches on a merged GP graph, -9.0 % warm per call at pop 1000 / rows 1024 and
#: -7 % at rows 16384, compile wall unchanged, identical numerics), while
#: ``default`` derives no flag at all and is byte-identical to the pre-knob build.
#: ``auto`` is the GPU-only enable rule: ``aggressive`` when the plan's compile
#: target is a GPU HAL (``rocm`` / ``cuda``, i.e. what
#: :func:`evogp.tree.batch_bands.resolve_band_policy` reports as the device kind)
#: AND the installed evoxc adapter knows the option, ``default`` otherwise --
#: llvm-cpu is flat-to-marginally-worse with it (117.2 -> 118.0 ms) and an older
#: evoxc raises on the unknown option, so ``auto`` never sends it there.
#: A spelling the installed adapter does not know degrades to ``default`` silently
#: (an explicit ``aggressive`` included): the feature is an optimisation, and a
#: missing evoxc must never turn a plan into a ``ValueError``/``BackendError``.
#: An unusable VALUE is still refused here, loudly.
DISPATCH_FORMATION_ENV = "EVOGP_BATCH_DISPATCH_FORMATION"
#: The ``auto`` spelling: let the target decide (see the block above).
DISPATCH_FORMATION_AUTO = "auto"
#: The two effective spellings a plan can carry (what the compile receives).
DISPATCH_FORMATION_AGGRESSIVE = "aggressive"
DISPATCH_FORMATION_DEFAULT = "default"
DEFAULT_DISPATCH_FORMATION = DISPATCH_FORMATION_AUTO
#: Every accepted spelling of :data:`DISPATCH_FORMATION_ENV`.
DISPATCH_FORMATIONS = (
    DISPATCH_FORMATION_AUTO,
    DISPATCH_FORMATION_AGGRESSIVE,
    DISPATCH_FORMATION_DEFAULT,
)
#: Spellings that mean "no derived flag" (``off`` is the env-friendly alias).
_DISPATCH_FORMATION_OFF = ("default", "off")
#: Smallest chunk worth a separate compile (``MIN_CHUNK_FUNCTIONS``): below it the
#: per-body fixed cost (one lower + one iree compile + one load) dominates, so a
#: chunked plan is only taken when ``n_functions >= MIN_CHUNK_FUNCTIONS`` and every
#: chunk keeps at least that many functions (``workers <= n // MIN_CHUNK_FUNCTIONS``).
#: Read through the MODULE object by :mod:`evogp.tree.batch_programs` so a test can
#: lower it to force chunking on a tiny fixture.
MIN_CHUNK_FUNCTIONS = 32
#: The two plan/graph variants: the value graph (``(pop, N, out_len)``) and the
#: SR-loss graph (one scalar per function, reduced IN-GRAPH).
VALUE_VARIANT = "value"
LOSS_VARIANT = "loss"
PLAN_VARIANTS = (VALUE_VARIANT, LOSS_VARIANT)
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_FLAT = ("none", "flat", "0")


# ---------------------------------------------------------------------------
# declines
# ---------------------------------------------------------------------------
class BatchUnsupported(RuntimeError):
    """The batch route cannot (safely) evaluate this population.

    Callers must fall back to the specialized engine
    (:func:`evogp.tree.evoxc_engine.decode_population` + ``evaluate`` /
    ``sr_fitness``), which owns the error contract for malformed trees.  The
    message carries the reason; the enumerated causes are listed in the module
    docstring ("Declines and the fallback rule"), and the two evoxc exception
    types a trace/merge can raise (``TraceError`` / ``TransformError``) are
    always converted into this one instead of escaping.
    """


# ---------------------------------------------------------------------------
# knob resolution (per call unless noted; the explicit argument always wins)
# ---------------------------------------------------------------------------
def _resolve_int(value: Any, env: str, default: int, minimum: int) -> int:
    """``explicit argument > environment > default`` for a bounded integer knob."""
    resolved = _env_int(env, default) if value is None else int(value)
    if resolved < minimum:
        raise ValueError(f"{env}={resolved} must be >= {minimum}")
    return resolved


def resolve_band_size(value: Optional[int] = None) -> int:
    """Resolve ``EVOGP_BATCH_BAND_SIZE`` (explicit argument wins).

    The band SIZE model turns a function count into ``K = ceil(n / size)`` bands.
    Read per call; anything below 1 raises ``ValueError`` naming the variable.
    """
    return _resolve_int(value, BAND_SIZE_ENV, DEFAULT_BAND_SIZE, 1)


def resolve_bands(value: Optional[Any] = None) -> Optional[Any]:
    """Resolve ``EVOGP_BATCH_BANDS`` into a band request or ``None``.

    The flat spellings (``none`` / ``flat`` / ``0``, and the integer ``0``) mean an
    explicit SINGLE band: they answer ``1``, so the merge is byte-identical to
    ``EVOGP_BATCH_BANDS=1`` (``band_sizes == [n_functions]``,
    ``band_strategy == "flat"``) instead of being handed to the band-size model.

    The ``auto`` spelling (any case, whitespace trimmed) answers the string
    :data:`BANDS_AUTO`, i.e. "let evoxc's cost model pick the partition"; the
    resulting band count is only known after the merge, so no count is resolved
    here (and :func:`resolve_band_count`, which is backend-agnostic and count-only,
    does not accept it -- :func:`evogp.tree.batch_bands.plan_band_spec` is the
    policy-aware resolver that does).

    ``None`` means "the band-size model decides" and is the answer for
    unset/empty only.  An integer ``K >= 1`` is returned as given;
    :func:`resolve_band_count` / :func:`evogp.tree.batch_bands.plan_band_spec`
    validate it against the function count (which is only known after decoding).
    Anything else raises ``ValueError`` naming the variable.
    """
    raw: Any = value if value is not None else os.environ.get(BANDS_ENV, "")
    if isinstance(raw, bool):  # bool is an int: reject instead of guessing
        raise ValueError(
            f"unsupported bands value {raw!r}: {BANDS_ENV} takes an integer band "
            f"count, the {BANDS_AUTO!r} spelling or one of {list(_FLAT)}"
        )
    if isinstance(raw, (int, np.integer)):
        count = int(raw)
        return 1 if count == 0 else count
    text = str(raw).strip().lower()
    if not text:
        return None
    if text == BANDS_AUTO:
        return BANDS_AUTO
    if text in _FLAT:
        return 1
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError(
            f"unsupported {BANDS_ENV}={raw!r}: expected an integer band count "
            f"(1 <= K <= n_functions), the {BANDS_AUTO!r} spelling or one of "
            f"{list(_FLAT)}"
        ) from exc
    return 1 if count == 0 else count


def resolve_chunk_bands(value: Optional[bool] = None) -> bool:
    """Resolve ``EVOGP_BATCH_CHUNK_BANDS`` (explicit argument wins, default off).

    Off (the default) keeps both knobs' documented per-plan meaning: a band
    request -- an explicit ``K``, a flat spelling or ``auto`` -- forces the
    single-executable compile path, and chunking keeps the band-size model per
    chunk.  On composes them: every chunk repeats the plan's band request (see
    :func:`evogp.tree.batch_bands.chunk_band_spec`).  ``switch_ops`` still always
    forces the single-body, flat merge.
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(CHUNK_BANDS_ENV, "").strip().lower()
    if not raw:
        return DEFAULT_CHUNK_BANDS
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"unsupported {CHUNK_BANDS_ENV}={raw!r}: batch_engine accepts "
        f"{list(_TRUE + _FALSE)} or an empty value"
    )


def resolve_switch_ops(value: Optional[bool] = None) -> bool:
    """Resolve ``EVOGP_BATCH_SWITCH_OPS`` (explicit argument wins; default off)."""
    if value is not None:
        return bool(value)
    raw = os.environ.get(SWITCH_OPS_ENV, "").strip().lower()
    if not raw:
        return False
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"unsupported {SWITCH_OPS_ENV}={raw!r}: batch_engine accepts "
        f"{list(_TRUE + _FALSE)} or an empty value"
    )


def resolve_alignment(value: Optional[str] = None) -> str:
    """Resolve ``EVOGP_BATCH_ALIGNMENT`` (explicit argument wins).

    Only meaningful with ``switch_ops`` (non-switched positions have one variant
    per skeleton by construction); validated always, so a typo is caught even on
    the default path.  Case-insensitive; ``variant`` / ``occurrence`` / ``auto``.
    """
    raw = value if value is not None else os.environ.get(ALIGNMENT_ENV, "")
    text = str(raw).strip().lower()
    if not text:
        text = DEFAULT_ALIGNMENT
    if text not in ALIGNMENTS:
        raise ValueError(
            f"unsupported alignment {text!r}: batch_engine supports "
            f"{list(ALIGNMENTS)} (set {ALIGNMENT_ENV!r} or pass alignment=)"
        )
    return text


def resolve_dispatch_formation(value: Optional[str] = None) -> str:
    """Resolve ``EVOGP_BATCH_DISPATCH_FORMATION`` into a REQUEST spelling.

    Precedence ``explicit argument > environment > default``; the answer is one of
    :data:`DISPATCH_FORMATIONS` (``auto`` / ``aggressive`` / ``default``), with the
    ``off`` spelling folded into ``default``.  ``auto`` (the default) is a request
    for the target policy, not an effective value: a caller turns it into what the
    compile receives with :func:`effective_dispatch_formation`, which is where the
    GPU-only enable rule and the evoxc feature detection live.

    Case-insensitive and whitespace-trimmed; anything else -- including a bool, a
    number or the ``none``/``flat`` spellings of other knobs -- raises
    ``ValueError`` naming the variable.
    """
    raw: Any = value if value is not None else os.environ.get(DISPATCH_FORMATION_ENV, "")
    if isinstance(raw, bool):
        raise ValueError(
            f"unsupported dispatch formation {raw!r}: {DISPATCH_FORMATION_ENV} takes "
            f"one of {list(DISPATCH_FORMATIONS)} or {list(_DISPATCH_FORMATION_OFF)}"
        )
    text = str(raw).strip().lower()
    if not text:
        return DEFAULT_DISPATCH_FORMATION
    if text in _DISPATCH_FORMATION_OFF:
        return DISPATCH_FORMATION_DEFAULT
    if text in DISPATCH_FORMATIONS:
        return text
    raise ValueError(
        f"unsupported {DISPATCH_FORMATION_ENV}={raw!r}: batch_engine accepts "
        f"{list(DISPATCH_FORMATIONS)} or {list(_DISPATCH_FORMATION_OFF)}"
    )


def effective_dispatch_formation(
    spelling: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    gpu: bool = False,
    supported: bool = False,
) -> str:
    """The spelling the merged compile actually receives: ``aggressive`` or ``default``.

    ``spelling`` is a REQUEST from :func:`resolve_dispatch_formation` (``None`` =
    resolve it here).  The rules, in order:

    * ``default`` / ``off`` -- ``default``, always (no derived flag);
    * ``auto`` -- ``aggressive`` only when all three hold: ``backend`` is ``iree``,
      ``gpu`` says the compile target is a GPU HAL (``rocm`` / ``cuda``; the caller
      reads it off :func:`evogp.tree.batch_bands.resolve_band_policy`, the one place
      the target kind is resolved) and ``supported`` says the installed evoxc
      adapter knows the option; ``default`` otherwise;
    * ``aggressive`` -- ``aggressive`` on ``iree`` with ``supported``, else
      ``default``: the flag is a GPU-codegen lever, and an evoxc without the option
      raises on it, so an explicit request degrades silently instead of failing the
      plan (only a malformed VALUE is an error, raised by
      :func:`resolve_dispatch_formation`).

    ``numpy`` is the reference interpreter: it documents options as accepted and
    ignored, so the route never puts the key there (it would change a memo key
    without changing a body).
    """
    request = resolve_dispatch_formation(spelling) if spelling is None else str(spelling)
    if request in _DISPATCH_FORMATION_OFF:
        request = DISPATCH_FORMATION_DEFAULT
    if request not in DISPATCH_FORMATIONS:
        request = DEFAULT_DISPATCH_FORMATION
    usable = str(backend or "").strip().lower() == "iree" and bool(supported)
    if request == DISPATCH_FORMATION_AGGRESSIVE:
        return DISPATCH_FORMATION_AGGRESSIVE if usable else DISPATCH_FORMATION_DEFAULT
    if request == DISPATCH_FORMATION_AUTO:
        return (
            DISPATCH_FORMATION_AGGRESSIVE
            if usable and bool(gpu)
            else DISPATCH_FORMATION_DEFAULT
        )
    return DISPATCH_FORMATION_DEFAULT


def resolve_min_pop(value: Optional[int] = None) -> int:
    """Resolve the population threshold ``EVOGP_BATCH_MIN_POP`` (argument wins)."""
    return _resolve_int(value, MIN_POP_ENV, DEFAULT_MIN_POP, 1)


def resolve_compile_workers(value: Optional[int] = None) -> int:
    """Resolve ``EVOGP_BATCH_COMPILE_WORKERS`` (explicit argument wins).

    ``0`` and ``1`` both mean "today's single-executable path verbatim": the
    deduped function list is merged into ONE graph and built as one body, so the
    plan's whole-graph fields (``graphs`` / ``merged`` / ``specs`` / ``report`` and
    ``executable``) keep their original meaning.  ``W > 1`` asks for ``W``
    contiguous chunks, subject to the forcing rules in
    :func:`evogp.tree.batch_programs._plan_host` -- a chunked plan also requires
    ``n_functions >= MIN_CHUNK_FUNCTIONS`` and is refused whenever ``switch_ops`` is
    in play or a band request (an explicit count, a flat spelling or ``auto``) is
    made without ``EVOGP_BATCH_CHUNK_BANDS``: those knobs must keep their documented
    per-plan meaning, and the band sweep runs on the single-body path unless the
    chunk-banded composition is opted into.

    Anything negative or non-integer raises ``ValueError`` naming the variable.
    """
    if value is None:
        raw = os.environ.get(COMPILE_WORKERS_ENV, "").strip()
        if not raw:
            return DEFAULT_COMPILE_WORKERS
        try:
            resolved = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"unsupported {COMPILE_WORKERS_ENV}={raw!r}: expected a "
                f"non-negative integer (0 and 1 both mean the single-executable "
                f"path, default {DEFAULT_COMPILE_WORKERS})"
            ) from exc
    else:
        if isinstance(value, bool):
            raise ValueError(
                f"unsupported {COMPILE_WORKERS_ENV}={value!r}: expected a "
                f"non-negative integer, not a bool"
            )
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"unsupported {COMPILE_WORKERS_ENV}={value!r}: expected a "
                f"non-negative integer"
            )
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported {COMPILE_WORKERS_ENV}={value!r}: expected a "
                f"non-negative integer"
            ) from exc
    if resolved < 0:
        raise ValueError(f"{COMPILE_WORKERS_ENV}={resolved} must be >= 0")
    return resolved


def resolve_band_count(
    n_functions: int,
    *,
    bands: Optional[Any] = None,
    band_size: Optional[int] = None,
    switch_ops: Optional[bool] = None,
) -> int:
    """The band count ``K`` this merge should use for ``n_functions`` functions.

    Precedence: ``switch_ops`` on -> ``K = 1`` (the flat merge; banding is never
    combined with op-switching), then an explicit ``bands`` count -- the flat
    spellings included, since :func:`resolve_bands` resolves them to ``1`` -- then
    the band-SIZE model ``ceil(n_functions / band_size)``.  The result is always
    clamped into ``1 <= K <= n_functions``; an explicit ``K`` outside that range
    raises ``ValueError`` naming ``EVOGP_BATCH_BANDS`` (it cannot be honoured, so
    silently clamping would hide a typo).

    The ``auto`` spelling is REFUSED here: it names no count before the merge
    (evoxc's cost model picks the partition at merge time), so a caller that wants
    it must go through :func:`evogp.tree.batch_bands.plan_band_spec`, which is the
    policy-aware resolver that can carry ``auto`` that far.
    """
    total = int(n_functions)
    if total < 1:
        raise ValueError(f"n_functions={total} must be >= 1")
    if resolve_switch_ops(switch_ops):
        return 1
    explicit = resolve_bands(bands)
    if isinstance(explicit, str):  # the ``auto`` spelling: no count to resolve here
        raise ValueError(
            f"{BANDS_ENV}={explicit!r} has no band count to resolve here: the "
            "automatic partition is chosen by evoxc's cost model at merge time, so "
            "the count only exists on the merged graph's batch_report (ask "
            f"{BANDS_ENV}=auto through plan_band_spec, or pass an explicit integer "
            f"K); this helper resolves / validates an explicit count only"
        )
    if explicit is not None:
        if not 1 <= explicit <= total:
            raise ValueError(
                f"{BANDS_ENV}={explicit} is outside 1..{total} (the number of "
                "merged functions)"
            )
        return explicit
    size = resolve_band_size(band_size)
    return max(1, min(total, -(-total // size)))


def _normalize_variant(variant: Any, per_row: bool) -> str:
    """The variant rules the batch route enforces (checked before any host work)."""
    text = str(variant).strip().lower()
    if text not in PLAN_VARIANTS:
        raise ValueError(
            f"unsupported plan variant {variant!r}: batch_engine supports "
            f"{list(PLAN_VARIANTS)}"
        )
    if text == LOSS_VARIANT and per_row:
        raise ValueError(
            "the loss variant implements sr_fitness only (per_row=False): a "
            "per_row=True loss would need one label row per tree, which is not "
            "the SR contract"
        )
    return text
