"""The batch route's cheap plan description (the ``describe_plan`` report seam).

:func:`describe_plan` answers "what would this plan build?" from the records alone --
nothing is decoded, traced or compiled -- and it is the report every test and every
benchmark record reads the merge shape from (``band_count`` / ``band_sizes`` /
``band_strategy``, the band policy's ``band_requested`` / ``band_clamped`` /
``band_note``, a chunked plan's per-body ``chunk_bands``, the plan's build shape and
the effective ``dispatch_formation``).

It lives beside rather than inside :mod:`evogp.tree.batch_engine` because it reads
only immutable :class:`evogp.tree.batch_programs.BatchPlan` fields: keeping it here
keeps the facade (plan memo + build) inside its size budget, and the facade
re-exports this name so ``evogp.tree.batch_engine.describe_plan`` stays the one
import path for callers and the benchmark harness.

"""

from __future__ import annotations

from typing import Any, Dict

from .batch_knobs import LOSS_VARIANT
from .batch_programs import BatchPlan

__all__ = ["describe_plan"]


def describe_plan(plan: BatchPlan) -> Dict[str, Any]:
    """A cheap, JSON-ish description of ``plan`` (for tests and a future runner).

    Reports the population size and the deduped function count, how many rows
    collide onto an earlier function, the merge shape (band count/strategy/sizes,
    emitted ops, class counters), the graph input shapes, the route
    (backend/device), and the plan's build shape: ``compile_workers`` (the chunk
    count actually used, ``1`` for the single-body plan), ``chunk_count`` /
    ``chunk_sizes`` (read off ``plan.chunks``, which holds one record per compiled
    body in both plan shapes) and the plan-side ``stage_ms`` split.  It reads the
    stored report only -- nothing is decoded, traced or compiled.

    A CHUNKED plan (``chunk_count > 1``) reports its whole-graph keys from the
    aggregate report: ``specs`` / ``spec_dtypes`` are empty (no single graph
    exists), ``band_sizes`` is empty and ``band_strategy`` / the class counters are
    ``None``; ``emitted_ops`` is the summed value and ``n_functions`` the global
    count.

    The band-policy keys are the cost model's report seam -- ``bands``, the two cost
    estimates, ``band_requested`` / ``band_requested_bands`` (the ``auto`` ``K`` before
    clamping), ``band_clamped`` / ``band_note``, ``band_policy`` and the per-body
    ``chunk_bands`` / ``chunk_band_strategies`` (missing -> ``None``/``False``/``{}``).

    The compile-option seam is reported too: ``dispatch_formation`` is the spelling
    the merged bodies are compiled with (``"default"`` or ``"aggressive"``, the value
    in the compile options and therefore in every memo / artifact key) and
    ``dispatch_formation_requested`` the request it came from (``"auto"`` /
    ``"aggressive"`` / ``"default"``), so a record can state what actually ran
    instead of what the environment asked for.
    """
    report = plan.report or {}
    shapes = [tuple(int(dim) for dim in spec.shape) for spec in plan.specs]
    return {
        "variant": plan.variant,
        "per_row": bool(plan.per_row),
        "use_mse": bool(plan.use_mse) if plan.variant == LOSS_VARIANT else None,
        "pop_size": int(plan.pop_size),
        "functions": len(plan.functions),
        "const_functions": sum(1 for program, _ in plan.functions if program is None),
        "row_index_collisions": int(plan.pop_size) - len(plan.functions),
        "input_len": int(plan.input_len),
        "output_len": int(plan.output_len),
        "n_rows": int(plan.n_rows),
        "band_count": int(plan.band_count),
        "band_sizes": list(report.get("band_sizes") or ()),
        "band_strategy": report.get("band_strategy"),
        "switch_ops": bool(plan.switch_ops),
        "alignment": plan.alignment,
        "emitted_ops": report.get("emitted_ops"),
        "value_classes": report.get("value_classes"),
        "shared_classes": report.get("shared_classes"),
        "stacked_constants": report.get("stacked_constants"),
        "output_muxes": report.get("output_muxes"),
        "n_functions": report.get("n_functions"),
        "bands": int(plan.band_count),
        "cost_estimate": report.get("cost_estimate"),
        "flat_cost_estimate": report.get("flat_cost_estimate"),
        "band_requested": plan.band_requested or report.get("band_requested"),
        "band_requested_bands": report.get("band_requested_bands"),
        "band_clamped": bool(plan.band_clamped or report.get("band_clamped")),
        "band_note": plan.band_note or report.get("band_note"),
        "band_policy": dict(plan.band_policy or {}),
        "chunk_bands": [int(chunk.band_count) for chunk in plan.chunks],
        "chunk_band_strategies": [
            chunk.report.get("band_strategy") for chunk in plan.chunks
        ],
        "dispatch_formation": plan.dispatch_formation,
        "dispatch_formation_requested": plan.dispatch_formation_requested,
        "compile_workers": int(plan.compile_workers),
        "chunk_count": len(plan.chunks),
        "chunk_sizes": [int(len(chunk.functions)) for chunk in plan.chunks],
        "stage_ms": dict(plan.stage_ms),
        "specs": shapes,
        "spec_dtypes": [str(spec.dtype) for spec in plan.specs],
        "backend": plan.backend_name,
        "device": str(plan.device),
    }
