"""The band/merge report of ONE merged batch-route plan (torch-free, evogp-free).

A batch record could state *whether* the merged graph ran, but not *what shape it merged into*: the
band count, the bands themselves, the merge strategy, the static cost model's two estimates, and -- the
part that makes the first two checkable -- the band-policy VERDICT (what was requested, the ``auto``
``K`` before clamping, whether the binding policy clamped it and why, the ``BandPolicy`` snapshot).  This
module is the ONE place that key list and the block-builder live, so ``worker.py`` / ``band_sweep.py`` /
``reuse_probe.py`` state the shape without carrying a second, subtly different key list.

The route-side contract is optional and getattr-probed: ``describe_plan(plan)`` is the cheap mapping view
(carrying ``band_count``/``chunk_count``/``chunk_bands`` and the un-clamped request), and the plan's OWN
``report`` (``merged.batch_report.as_dict()``) is where ``bands``, the two cost-model estimates and the
per-body ``cost_estimate``/``flat_cost_estimate`` live.  Nothing here imports ``evogp`` or ``torch``: it
only receives a plan object and/or the mapping a caller already has.

**Null, never zero.**  A field neither source carries reads ``None`` (``available: false`` when nothing
could be read) plus a ``note`` -- never a fabricated ``0``, ``False`` or ``[]``: "no bands", "not
clamped" and "nobody said" must never look alike in a record.  ``band_clamped`` is coerced with ``bool``
only when it is PRESENT, and ``band_policy`` is COPIED so a caller can never mutate the plan's snapshot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: The band/merge fields a record keeps per MERGED PLAN.  ``describe_plan`` exposes
#: most of them, but not ``bands`` and not the two cost-model estimates: those live
#: in the plan's OWN report (``plan.report`` = ``merged.batch_report.as_dict()``),
#: which is the only place ``cost_estimate`` / ``flat_cost_estimate`` are written --
#: and they stay ``None`` unless the static cost model ran
#: (``EVOGP_BATCH_BANDS=auto``), which is a fact, not a defect.
#: The trailing six keys are the policy VERDICT: what was requested, the ``auto`` ``K``
#: before clamping, whether the binding policy clamped it and why, the ``BandPolicy``
#: snapshot, and the per-body band counts of a chunked plan (which only
#: ``describe_plan`` carries) -- i.e. "what the cost model chose" next to the measured
#: shape.  Appended LAST so the older eight keep their order.
BAND_REPORT_KEYS: Tuple[str, ...] = (
    "bands",
    "band_sizes",
    "band_strategy",
    "cost_estimate",
    "flat_estimate",
    "n_functions",
    "emitted_ops",
    "band_count",
    "band_requested",
    "band_requested_bands",
    "band_clamped",
    "band_note",
    "band_policy",
    "chunk_bands",
)
#: Why the two cost estimates are usually null here (the row is machine-readable too).
BAND_COST_NOTE = (
    "cost_estimate / flat_estimate are null because the static cost model did not run: "
    "only EVOGP_BATCH_BANDS=auto runs it, and a pinned band SIZE (or the flat merge) "
    "never does -- null (NOT 0) means 'not measured', never 'the estimate was zero'.  "
    "When auto DID run, both estimates are populated and band_requested reads 'auto'."
)
BAND_ABSENT_NOTE = (
    "no merged-plan description was available (no describe_plan mapping and no plan "
    "carrying a report: the plan memo / describe_plan was not reachable), so every band "
    "field is null (NOT zero)."
)


def _plan_report(plan: Any) -> Dict[str, Any]:
    """``plan.report`` when it is a dict, else ``{}`` (a mapping argument has none).

    Never raises: an attribute that raises, an absent attribute or a report of some
    other type is reported as "no report", which is what a caller records.
    """
    if isinstance(plan, dict):
        return {}
    try:
        report = getattr(plan, "report", None)
    except Exception:  # noqa: BLE001 - an attribute that raises reads as absent
        return {}
    return report if isinstance(report, dict) else {}


def _band_sizes(value: Any) -> Optional[List[int]]:
    """``band_sizes`` as a list of ints (a non-list reads as absent, never as ``[]``)."""
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> Optional[bool]:
    """A PRESENT flag as ``bool``; an ABSENT one stays ``None``, never a fabricated False."""
    return None if value is None else bool(value)


def _copied_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """A shallow COPY of a report mapping (so a caller cannot mutate the plan), else None.

    A non-mapping reads as absent, the same rule :func:`_band_sizes` applies to a
    non-list: the record states "this plan carried no policy snapshot", never ``{}``.
    """
    return dict(value) if isinstance(value, dict) else None


def band_report(plan_or_described: Any = None, described: Any = None) -> Dict[str, Any]:
    """The band/merge shape of ONE merged plan, next to the cost model's own verdict.

    Accepts a ``describe_plan``-style mapping, a ``BatchPlan`` (whose ``report``
    attribute is ``merged.batch_report.as_dict()``), or both -- ``described`` is the
    mapping view when the caller already has it.  The plan's OWN report wins wherever
    it speaks, and that is what keeps a CHUNKED plan honest: its aggregate report sets
    ``bands`` / ``band_sizes`` / ``band_strategy`` to ``None`` (no single merged graph
    exists), and that null is recorded as null rather than replaced by a fabricated
    ``[]`` or ``1``.  ``cost_estimate`` / ``flat_estimate`` come from that report alone.

    It also carries the band-policy VERDICT -- ``band_requested`` (the request
    spelling: flat/explicit/size/auto), ``band_requested_bands`` (the ``auto`` ``K``
    BEFORE clamping, null when not auto), ``band_clamped`` / ``band_note`` (whether the
    binding policy lowered the count, and why) and ``band_policy`` (a COPY of the
    ``BandPolicy`` snapshot, so a caller can never mutate the plan) -- plus the
    per-body ``chunk_bands`` of a chunked plan, which only the ``describe_plan`` view
    carries.  That is what lets a record show "what the cost model chose" next to
    "what measured best"; a plan whose report predates those keys reads ``None``
    everywhere, ``band_clamped`` included (an absent clamp flag is NOT ``False``).

    Every field is ``None`` when neither source carries it (never ``0``), ``available``
    is true only when at least one field could be read, and nothing here imports
    ``torch``/``evogp`` or raises: this is compile-free introspection of a plan the route
    has already stored, so a missing memo costs a block, never a record.
    ``chunk_count`` rides along from the ``describe_plan`` view (it lives there, not in
    the report) so a chunked plan's nulls can be read for what they are.
    """
    mapping = (
        described
        if isinstance(described, dict)
        else (plan_or_described if isinstance(plan_or_described, dict) else {})
    )
    report = _plan_report(plan_or_described)

    def pick(key: str, *fallback: str) -> Any:
        """The report's own value where the key exists, else the describe_plan view's."""
        if key in report:
            return report[key]
        for name in (key,) + fallback:
            if name in mapping:
                return mapping[name]
        return None

    values: Dict[str, Any] = {
        "bands": pick("bands"),
        # an empty band_sizes list is what the aggregate report of a chunked plan means:
        # no single merge shape, never 0 bands
        "band_sizes": _band_sizes(pick("band_sizes")),
        "band_strategy": pick("band_strategy"),
        "cost_estimate": pick("cost_estimate"),
        "flat_estimate": pick("flat_cost_estimate", "flat_estimate"),
        "n_functions": pick("n_functions"),
        "emitted_ops": pick("emitted_ops"),
        "band_count": pick("band_count"),
        # the policy verdict: the request spelling, the auto K before clamping, the
        # clamp flag/note and the policy snapshot (copied, never the plan's own dict)
        "band_requested": pick("band_requested"),
        "band_requested_bands": pick("band_requested_bands"),
        # a present flag is coerced, an ABSENT one stays None: "not clamped" and
        # "nobody said" must not look alike
        "band_clamped": _opt_bool(pick("band_clamped")),
        "band_note": pick("band_note"),
        "band_policy": _copied_mapping(pick("band_policy")),
        # per-body band counts of a CHUNKED plan (describe_plan only; null when absent)
        "chunk_bands": _band_sizes(pick("chunk_bands")),
    }
    if report and mapping:
        source = "plan-report+described"
    elif report:
        source = "plan-report"
    elif mapping:
        source = "described"
    else:
        source = "absent"
    notes: List[str] = []
    if source == "absent":
        notes.append(BAND_ABSENT_NOTE)
    elif source == "plan-report":
        notes.append(
            "only the plan's own report was readable, so band_count / chunk_count are null "
            "wherever only the describe_plan view carries them."
        )
    if source != "absent" and values["cost_estimate"] is None and values["flat_estimate"] is None:
        notes.append(BAND_COST_NOTE)
    block: Dict[str, Any] = {
        "available": any(value is not None for value in values.values()),
        "source": source,
        "note": " ".join(notes) or None,
        "chunk_count": mapping.get("chunk_count"),
        "report_keys": sorted(str(key) for key in report),
    }
    for key in BAND_REPORT_KEYS:
        block[key] = values.get(key)
    return block
