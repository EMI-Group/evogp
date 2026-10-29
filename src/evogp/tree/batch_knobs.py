"""The batch route's knob vocabulary, variant rules and decline type.

One of the three modules the opt-in batch evaluation route lives in:

* **this module** -- the ``EVOGP_BATCH_*`` constants, the ``resolve_*`` helpers
  that read them per call, the ``value`` / ``loss`` variant spellings with
  :func:`_normalize_variant`, and :class:`BatchUnsupported`.
* :mod:`evogp.tree.batch_programs` -- host-side program handling: the input /
  label conversions, the per-function graph bodies, the structural dedup, the
  trace and the merge, plus the :class:`BatchPlan` / :class:`BatchExecutable`
  records.
* :mod:`evogp.tree.batch_engine` -- the PUBLIC facade (import that one, not the
  two siblings): the two front tables, :func:`plan_batch` / :func:`build_batch` /
  :func:`evaluate_batch` / :func:`sr_fitness_batch` / :func:`describe_plan` and
  :func:`batch_cache_stats`, which re-exports everything defined here.

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
#: bands, clamped to ``1 <= K <= n``.  32 measured as a sound default: it keeps
#: the per-band variant count low without splitting a small population into
#: bands that each re-pay their own input slice and output concatenate.
BAND_SIZE_ENV = "EVOGP_BATCH_BAND_SIZE"
DEFAULT_BAND_SIZE = 32
#: Explicit band count (``EVOGP_BATCH_BANDS``): an int ``K`` with
#: ``1 <= K <= n_functions``, or one of the flat spellings ``none`` / ``flat`` /
#: ``0``.  Unset/empty means "use the band-size model above".
BANDS_ENV = "EVOGP_BATCH_BANDS"
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


def resolve_bands(value: Optional[Any] = None) -> Optional[int]:
    """Resolve ``EVOGP_BATCH_BANDS`` into an explicit band count or ``None``.

    ``None`` means "the band-size model decides" -- the answer for unset/empty
    AND for the explicit flat spellings (``none`` / ``flat`` / ``0``), which both
    describe a single-band (flat) merge.  An integer ``K`` is returned as given;
    :func:`resolve_band_count` validates it against the function count (which is
    only known after decoding).  Anything else raises ``ValueError`` naming the
    variable.
    """
    raw: Any = value if value is not None else os.environ.get(BANDS_ENV, "")
    if isinstance(raw, bool):  # bool is an int: reject instead of guessing
        raise ValueError(
            f"unsupported bands value {raw!r}: {BANDS_ENV} takes an integer band "
            f"count or one of {list(_FLAT)}"
        )
    if isinstance(raw, (int, np.integer)):
        count = int(raw)
        return None if count == 0 else count
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in _FLAT:
        return None
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError(
            f"unsupported {BANDS_ENV}={raw!r}: expected an integer band count "
            f"(1 <= K <= n_functions) or one of {list(_FLAT)}"
        ) from exc
    return None if count == 0 else count


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


def resolve_min_pop(value: Optional[int] = None) -> int:
    """Resolve the population threshold ``EVOGP_BATCH_MIN_POP`` (argument wins)."""
    return _resolve_int(value, MIN_POP_ENV, DEFAULT_MIN_POP, 1)


def resolve_band_count(
    n_functions: int,
    *,
    bands: Optional[Any] = None,
    band_size: Optional[int] = None,
    switch_ops: Optional[bool] = None,
) -> int:
    """The band count ``K`` this merge should use for ``n_functions`` functions.

    Precedence: ``switch_ops`` on -> ``K = 1`` (the flat merge; banding is never
    combined with op-switching), then an explicit ``bands`` count, then the
    band-SIZE model ``ceil(n_functions / band_size)``.  The result is always
    clamped into ``1 <= K <= n_functions``; an explicit ``K`` outside that range
    raises ``ValueError`` naming ``EVOGP_BATCH_BANDS`` (it cannot be honoured, so
    silently clamping would hide a typo).
    """
    total = int(n_functions)
    if total < 1:
        raise ValueError(f"n_functions={total} must be >= 1")
    if resolve_switch_ops(switch_ops):
        return 1
    explicit = resolve_bands(bands)
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
