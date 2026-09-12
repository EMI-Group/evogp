"""Device-resident hot-path check + measurement for the dense evaluator.

Runnable script, deliberately NOT a pytest module (no ``test_`` prefix, no pytest
import).  It is the evidence battery for the **device-resident** work on
:mod:`evogp.tree.dense_engine`: the SR loss is folded into the compiled dense
graph, every host<->device crossing the hot path makes is *counted*, and the
invariants that make the path device-resident are asserted.

What it instruments
-------------------
:class:`Traffic` monkeypatches the functions through which a dense call can cross
the host boundary and records, per call, the **caller's function name**, the
payload name (where the site has one) and the payload shape:

=================================== ========================== ==========================
patched site                        direction                  what it sees
=================================== ========================== ==========================
``dense_engine._as_numpy``          torch -> host              ``inputs`` / ``x`` / ``labels`` / the three population tensors
``etl.from_numpy``                  host -> ETL tensor         every upload the library asks for (device backends)
``dense_engine._pad_rows``          host only                  every chunk payload it PREPARES (all backends)
``etl.core.Tensor.numpy``           ETL -> host                every chunk readback (filtered by caller frame)
``torch.from_numpy``                host -> torch tensor       the host-side tensor assembly
``etl.run``                         graph invocation           every compiled-graph call (one per chunk is the invariant)
=================================== ========================== ==========================

``etl.from_numpy`` is the upload site, so the "each distinct payload is moved at
most once per call" invariant is only exercisable on a device backend; the
``_pad_rows`` counter measures the same redundancy on the numpy backend, where
the payload never leaves the host (a payload is uploaded only after being
prepared, and the two counters agree on chunk multiplicity).  ``etl.core.Tensor.numpy``
is ALSO called thousands of times inside the numpy interpreter's own tensor
arithmetic, so the readback counter is filtered by caller frame -- and the
``seam`` stage carries a positive control (the same filter must see ``run_plan``'s
readbacks) so a filter that silently stopped matching cannot pass.

The counters are caller-frame NAMES for the readback / upload / assembly paths,
so they also work on the pre-refactor code -- which is what makes this script
usable as the "before" evidence instead of crashing on a missing symbol.
Checks
------
``policy``   ``dynamic`` is accepted as an explicit alias of ``dense`` in
             ``EVOGP_EVAL_MODE`` / ``eval_mode=`` (and an unknown value still
             raises ``ValueError``); the alias is never gated by the ``auto`` rows
             rule.
``hotpath``  one FORCED-CHUNKED dense ``Forest.SR_fitness`` call (small
             ``EVOGP_DENSE_BUDGET_MB``, so ``p_chunks, n_chunks > 1``) must:
             convert ``inputs`` exactly once and ``labels`` exactly once, read
             back only ``(P,)-``scale chunk losses (never a full prediction),
             create no torch tensor from a host array larger than ``pop_size``,
             and upload each distinct payload at most once per call (``x`` and
             ``labels`` once per ``n``-chunk, the seven node columns once per
             ``p``-chunk).
``parity``   forced-dense vs forced-specialized through the public API (MSE/MAE,
             NaN/+-inf labels and inputs, a padded last chunk whose row count is
             not a multiple of the chunk size, the pop=1 ``Tree`` path, and the
             untouched value path), plus the chunked result against the
             single-chunk result.
``seam``     the readback-free run helper (``dense_engine._iter_plan_chunks``)
             exists, yields one raw (un-read) result per chunk after exactly one
             compiled-graph invocation, performs ZERO readbacks (with a positive
             control showing the counter does see ``run_plan``'s readbacks),
             moves each distinct payload (``7`` columns per ``p``-chunk, one
             ``x`` per ``n``-chunk) exactly once per call -- measured as
             ``etl.from_numpy`` uploads on a device backend (``--backend iree``)
             and as ``_pad_rows`` preparations on numpy -- and its chunks
             assemble into the same host array ``run_plan`` returns (bit-exact,
             NaN/±inf masks included).
``measure``  (``--measure``) warm per-call medians for the dense SR path at the
             ``P/N`` grid points, the pop=1 ``Tree`` path and the value path
             (``Forest.forward``), each with the per-call traffic summary -- the
             before/after numbers of ``device_resident_findings.md``.

Usage (repo root, project venv)::

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      <venv>/bin/python src/evogp/tree/device_resident_check.py
    ... device_resident_check.py --measure --json /tmp/device_resident.json
    ... device_resident_check.py --backend iree --gpu 0 --cases 16x64

Exit code 0 = every check passed, 1 = at least one failed, 2 = hard error.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import etl  # noqa: E402

from evogp.tree import Forest  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402
from evogp.tree.utils import FUNCS_NAMES, Func, NType  # noqa: E402

# Small helpers shared with the integration battery (same directory): the report
# object, the float32 comparison policy and the contract checks.  Reusing them
# keeps every dense battery's pass/fail semantics identical.
from evogp.tree.dense_integration_check import (  # noqa: E402
    Report,
    check_contract,
    check_writable,
    compare,
    random_population,
    set_mode,
)

_RTOL = 1e-4
_ATOL = 1e-4

#: the library's default chunking budget (MiB): the "unchunked by default" regime
DEFAULT_BUDGET_MB = float(getattr(D, "DEFAULT_BUDGET_MB", 2048))

#: callers that make up the dense run machinery's READBACK path (pre-refactor:
#: ``run_plan`` reads every chunk back itself; post-refactor: the single helper
#: ``_to_host`` inside the chunk iterator)
_READBACK_FRAMES = ("run_plan", "run_loss_plan", "_to_host", "_iter_plan_chunks")
#: callers that make up its UPLOAD path (pre-refactor: ``run_plan`` uploads per
#: (p0, n0) chunk; post-refactor: the memoizing ``payload`` closure)
_UPLOAD_FRAMES = ("run_plan", "columns", "data", "_upload", "_iter_plan_chunks")
#: callers that invoke the compiled graph (one call per chunk is the invariant)
_RUN_FRAMES = ("run_plan", "run_loss_plan", "_iter_plan_chunks")
#: callers that make up its TORCH-ASSEMBLY path (the host -> torch tensor step)
_DENSE_FRAMES = (
    "evaluate_dense",
    "sr_fitness_dense",
    "run_plan",
    "run_loss_plan",
    "_to_host",
    "_iter_plan_chunks",
)

def _caller(depth: int = 2) -> str:
    """Name of the function that called the PATCHED SITE (one frame above the wrapper)."""
    return sys._getframe(depth).f_code.co_name


# ---------------------------------------------------------------------------
# traffic counters
# ---------------------------------------------------------------------------


class Traffic:
    """Every host<->device crossing of one dense call, by site and caller.

    A context manager: it patches the conversion/readback helpers for the
    duration of the ``with`` block and restores them on exit (exception
    included), so nested measurements are safe.  Each record's first element is
    ALWAYS the caller's function name.
    """

    def __init__(self) -> None:
        self.as_numpy: List[Tuple[str, str, int]] = []
        self.from_numpy: List[Tuple[str, Tuple[int, ...]]] = []
        self.tensor_numpy: List[Tuple[str, Tuple[int, ...]]] = []
        self.torch_from_numpy: List[Tuple[str, Tuple[int, ...]]] = []
        self.pad_rows: List[Tuple[str, Tuple[int, ...]]] = []
        self.run_calls: List[Tuple[str, str]] = []

    # -- patching ----------------------------------------------------------
    def __enter__(self) -> "Traffic":
        self._originals = (
            D._as_numpy,
            etl.from_numpy,
            etl.core.Tensor.numpy,
            torch.from_numpy,
            D._pad_rows,
            etl.run,
        )
        (
            original_as_numpy,
            original_from_numpy,
            original_numpy,
            original_torch,
            original_pad_rows,
            original_run,
        ) = self._originals

        def as_numpy(tensor, dtype, name):
            array = original_as_numpy(tensor, dtype, name)
            self.as_numpy.append((_caller(), str(name), int(array.size)))
            return array

        def from_numpy(array):
            self.from_numpy.append((_caller(), tuple(np.shape(array))))
            return original_from_numpy(array)

        def tensor_numpy(self_: Any, *args: Any, **kwargs: Any):
            result = original_numpy(self_, *args, **kwargs)
            self.tensor_numpy.append((_caller(), tuple(self_.shape)))
            return result

        def torch_from_numpy(array, *args: Any, **kwargs: Any):
            self.torch_from_numpy.append((_caller(), tuple(np.shape(array))))
            return original_torch(array, *args, **kwargs)

        def pad_rows(array, rows):
            self.pad_rows.append((_caller(), tuple(np.shape(array))))
            return original_pad_rows(array, rows)

        def run(executable, *args, **kwargs):
            self.run_calls.append((_caller(), type(executable).__name__))
            return original_run(executable, *args, **kwargs)

        D._as_numpy = as_numpy
        etl.from_numpy = from_numpy
        etl.core.Tensor.numpy = tensor_numpy
        torch.from_numpy = torch_from_numpy
        D._pad_rows = pad_rows
        etl.run = run
        return self

    def __exit__(self, *exc: Any) -> None:
        (
            D._as_numpy,
            etl.from_numpy,
            etl.core.Tensor.numpy,
            torch.from_numpy,
            D._pad_rows,
            etl.run,
        ) = self._originals
        return None
    # -- views -------------------------------------------------------------
    @staticmethod
    def _filtered(records: Sequence[Any], frames: Sequence[str]) -> List[Any]:
        return [record for record in records if record[0] in frames]

    @property
    def readbacks(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Chunk readbacks issued by the dense run machinery."""
        return self._filtered(self.tensor_numpy, _READBACK_FRAMES)

    @property
    def uploads(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Host -> device uploads issued by the dense run machinery."""
        return self._filtered(self.from_numpy, _UPLOAD_FRAMES)

    @property
    def torch_creations(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Host -> torch tensor creations issued by the dense path."""
        return self._filtered(self.torch_from_numpy, _DENSE_FRAMES)

    @property
    def runs(self) -> List[Tuple[str, str]]:
        """Compiled-graph invocations issued by the dense run machinery."""
        return [record for record in self.run_calls if record[0] in _RUN_FRAMES]

    def preparations(self, last_dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """``_pad_rows`` payload PREPARATIONS whose trailing axis is ``last_dim``.

        Backend-independent: the numpy backend never uploads, so on numpy the
        per-payload preparation count is the observable proxy for "this payload
        is moved once per call" (a payload re-prepared per ``(p0, n0)`` chunk is
        also re-uploaded per ``(p0, n0)`` chunk on a device backend).  The
        counter deliberately does NOT filter by caller frame: ``_pad_rows`` is
        private to ``dense_engine``, whose only caller is the dense run
        machinery, and the preparation step now sits behind the payload memo
        (``_chunk_rows`` / ``_keep_mask``), so a frame-name filter would silently
        stop counting the very calls it is meant to count.
        """
        return [
            record
            for record in self.pad_rows
            if record[1] and record[1][-1] == last_dim
        ]
    def conversions(self, *names: str) -> List[Tuple[str, str, int]]:
        """``_as_numpy`` (torch -> host) calls whose payload name is in ``names``."""
        wanted = set(names)
        return [record for record in self.as_numpy if record[1] in wanted]

    def upload_counts(self) -> Dict[Tuple[int, ...], int]:
        """Upload multiplicity by shape (the "uploaded once per call" evidence)."""
        counts: Dict[Tuple[int, ...], int] = {}
        for _, shape in self.uploads:
            counts[shape] = counts.get(shape, 0) + 1
        return counts

    def summary(self, pop_size: int) -> Dict[str, Any]:
        """Compact, JSON-friendly per-call traffic summary."""
        readback_elements = [int(np.prod(shape)) for _, shape in self.readbacks]
        creations = [int(np.prod(shape)) for _, shape in self.torch_creations]
        return {
            "pop_size": int(pop_size),
            "host_reads": {
                name: len(self.conversions(name)) for name in ("inputs", "x", "labels")
            },
            "as_numpy_calls": len(self.as_numpy),
            "readbacks": len(self.readbacks),
            "readback_shapes": [list(shape) for _, shape in self.readbacks],
            "readback_max_elements": max(readback_elements, default=0),
            "readback_total_elements": sum(readback_elements),
            "uploads": len(self.uploads),
            "upload_shapes": {
                str(shape): count
                for shape, count in sorted(self.upload_counts().items())
            },
            "upload_total_elements": sum(
                int(np.prod(shape)) for _, shape in self.uploads
            ),
            "torch_from_host": len(self.torch_creations),
            "torch_from_host_shapes": [
                list(shape) for _, shape in self.torch_creations
            ],
            "torch_from_host_max_elements": max(creations, default=0),
            "etl_runs": len(self.runs),
        }


def _raises(exc_type: type, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """True iff ``fn(*args)`` raised exactly ``exc_type``."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception:
        return False
    return False


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args)``'s result, or an ``<ExcType: msg>`` marker instead of raising.

    The battery must run on the PRE-refactor code too (that is what makes it the
    "before" evidence), where ``dynamic`` is not a known mode yet.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # surfaced by the caller as a mismatch
        return f"<{type(exc).__name__}: {exc}>"


def _safe_mode(value: str) -> Any:
    return _safe(D.resolve_eval_mode, value)


def _env_mode(value: str) -> Any:
    """``resolve_eval_mode()`` with ``EVOGP_EVAL_MODE`` set to ``value``."""
    previous = os.environ.get(D.EVAL_MODE_ENV)
    os.environ[D.EVAL_MODE_ENV] = value
    try:
        return _safe(D.resolve_eval_mode)
    finally:
        if previous is None:
            os.environ.pop(D.EVAL_MODE_ENV, None)
        else:
            os.environ[D.EVAL_MODE_ENV] = previous


class _backend:
    """Set ``EVOGP_ETL_BACKEND`` (+ the IREE device index) for the block.

    The seam stage needs the backend the ENGINE resolves (``_resolve_backend``
    reads the env), not just the mode switch, so it flips the env the same way
    ``--measure`` does.  ``None`` / ``"numpy"`` = the default backend (unset).
    """

    def __init__(self, name: Optional[str], gpu: Optional[int] = None) -> None:
        self.name = None if name in (None, "numpy") else str(name)
        self.gpu = gpu

    def __enter__(self) -> "_backend":
        self.previous = (
            os.environ.get("EVOGP_ETL_BACKEND"),
            os.environ.get("EVOGP_ETL_IREE_DEVICE"),
        )
        if self.name is None:
            os.environ.pop("EVOGP_ETL_BACKEND", None)
        else:
            os.environ["EVOGP_ETL_BACKEND"] = self.name
            if self.gpu is not None:
                os.environ["EVOGP_ETL_IREE_DEVICE"] = str(int(self.gpu))
        return self

    def __exit__(self, *exc: Any) -> None:
        backend, device = self.previous
        for key, value in (
            ("EVOGP_ETL_BACKEND", backend),
            ("EVOGP_ETL_IREE_DEVICE", device),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return None


class _budget:
    """Force a chunked plan for the duration of the block (reads the env per call)."""

    def __init__(self, budget_mb: Optional[float]) -> None:
        self.budget_mb = budget_mb

    def __enter__(self) -> "_budget":
        self.previous = os.environ.get(D.BUDGET_MB_ENV)
        if self.budget_mb is None:
            os.environ.pop(D.BUDGET_MB_ENV, None)
        else:
            os.environ[D.BUDGET_MB_ENV] = repr(float(self.budget_mb))
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.previous is None:
            os.environ.pop(D.BUDGET_MB_ENV, None)
        else:
            os.environ[D.BUDGET_MB_ENV] = self.previous
        return None


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def fixture(
    pop: int, rows: int, *, input_len: int = 3, seed: int = 0
) -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    """A random forest plus SR inputs/labels on the CUDA device."""
    forest = random_population(seed, pop, 16, input_len, 0.35, FUNCS_NAMES, 3)
    torch.manual_seed(10_000 + seed)
    inputs = torch.randn(rows, input_len, device="cuda")
    labels = torch.randn(rows, 1, device="cuda")
    return forest, inputs, labels


def cap_chain(
    longest: int, row_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One tree of EXACTLY ``longest`` nodes: ``longest-1`` SIN nodes over a VAR.

    A right-leaning chain, so the declared tree length is deterministic -- which
    is what the trace-cap checks need (a random population's REALIZED length is
    not).  Returned as the raw ``(node_value, node_type, subtree_size)`` triple
    of a one-row population, in the ``(pop, max_tree_len)`` layout
    ``prepare_dense`` consumes.
    """
    values = np.zeros((1, row_len), dtype=np.float32)
    types = np.zeros((1, row_len), dtype=np.int16)
    sizes = np.ones((1, row_len), dtype=np.int16)
    for index in range(longest - 1):
        types[0, index] = int(NType.UFUNC)
        values[0, index] = float(Func.SIN)
        sizes[0, index] = longest - index
    types[0, longest - 1] = int(NType.VAR)
    values[0, longest - 1] = 0.0
    sizes[0, longest - 1] = 1
    return (
        torch.from_numpy(values),
        torch.from_numpy(types),
        torch.from_numpy(sizes),
    )


def _cap_attempt(
    longest: int,
    row_len: int,
    *,
    backend: Optional[str] = None,
    output_len: int = 1,
) -> Tuple[Any, Optional[str], List[str]]:
    """``prepare_dense`` on one chain, with its ``RuntimeWarning``s captured.

    Returns ``(DensePrep or None, DenseUnsupported reason or None, warning
    messages)``.  The ``EVOGP_DENSE_MAX_TREE_LEN`` dedupe flag is process-global,
    so the caller resets ``D._TRACE_CAP_WARNED`` when it counts warnings.
    """
    values, types, sizes = cap_chain(longest, row_len)
    kwargs: Dict[str, Any] = {}
    if backend is not None:
        kwargs["backend"] = backend
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        prep: Any = None
        reason: Optional[str] = None
        try:
            prep = D.prepare_dense(values, types, sizes, 1, output_len, **kwargs)
        except D.DenseUnsupported as exc:
            reason = str(exc)
    warnings_seen = [
        str(item.message)
        for item in recorded
        if issubclass(item.category, RuntimeWarning)
    ]
    return prep, reason, warnings_seen


def plan_for(
    forest: Forest,
    rows: int,
    *,
    budget_mb: Optional[float],
    variant: str = "value",
) -> Any:
    """``plan_dense`` for this fixture (the ``variant`` argument is post-refactor)."""
    args = (
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        rows,
    )
    kwargs: Dict[str, Any] = {
        "per_row": False,
        "budget_mb": budget_mb,
        "backend": os.environ.get("EVOGP_ETL_BACKEND"),
    }
    try:
        return D.plan_dense(*args, variant=variant, **kwargs)
    except TypeError:  # pre-refactor plan_dense: no variant parameter
        return D.plan_dense(*args, **kwargs)


def _plan_line(plan: Any) -> str:
    return (
        f"p_chunks={plan.p_chunks} n_chunks={plan.n_chunks} "
        f"p_chunk={plan.p_chunk} n_chunk={plan.n_chunk} "
        f"L={plan.trace_len} pop={plan.pop_size} rows={plan.n_rows}"
    )


#: budgets the tuned search walks (MiB).  ``_chunk_shape`` only ever produces
#: ``p_chunks > 1 and n_chunks > 1`` when the per-chunk element budget falls
#: below ``pop_size`` (then ``n_chunk == 1``); a merely row-chunked plan
#: (``p_chunks == 1``) is the regime where a full-prediction readback is
#: distinguishable from a ``(P,)``-scale one, so both are looked for.
_CANDIDATE_BUDGETS = (
    0.0004,
    0.0006,
    0.0008,
    0.0012,
    0.002,
    0.004,
    0.008,
    0.016,
    0.05,
    0.2,
    1.0,
)


def _tuned_budget(
    forest: Forest,
    rows: int,
    variant: str,
    predicate: Callable[[Any], bool],
) -> Tuple[Optional[float], Any]:
    """First candidate budget whose plan satisfies ``predicate`` (else ``(None, None)``)."""
    for budget in _CANDIDATE_BUDGETS:
        plan = plan_for(forest, rows, budget_mb=budget, variant=variant)
        if predicate(plan):
            return budget, plan
    return None, None


# ---------------------------------------------------------------------------
# stage: policy (the `dynamic` alias)
# ---------------------------------------------------------------------------


def stage_policy(args: Any, report: Report) -> None:
    print("== stage policy: `dynamic` as an explicit alias of `dense` ==")
    aliases = getattr(D, "EVAL_MODE_ALIASES", {})
    mode = _safe_mode("dynamic")
    report.add(
        "resolve_eval_mode('dynamic') -> 'dense'",
        mode == "dense",
        f"got {mode!r}",
    )
    spaced = _safe_mode("  DyNaMic ")
    report.add(
        "the alias is case/whitespace insensitive",
        spaced == "dense",
        f"got {spaced!r}",
    )
    env_mode = _env_mode("dynamic")
    report.add(
        f"{D.EVAL_MODE_ENV}=dynamic resolves to dense",
        env_mode == "dense",
        f"got {env_mode!r}",
    )
    alias_attempt = _safe(
        D.should_use_dense, 4, eval_mode="dynamic", backend="numpy"
    )
    literal_attempt = _safe(
        D.should_use_dense, 4, eval_mode="dense", backend="numpy"
    )
    report.add(
        "should_use_dense(eval_mode='dynamic') attempts dense on numpy",
        alias_attempt is True and literal_attempt is True,
        f"dynamic -> {alias_attempt!r}, dense -> {literal_attempt!r} "
        "(an explicit alias is the user's opt-in, exactly like 'dense')",
    )
    decline = _safe(D.auto_rows_decline, 10**6, eval_mode="dynamic")
    report.add(
        "auto_rows_decline() never gates an explicit dynamic",
        decline is None,
        f"got {decline!r}",
    )
    report.add(
        "an unknown mode still raises ValueError",
        _raises(ValueError, D.resolve_eval_mode, "dynamic2")
        and _raises(ValueError, D.resolve_eval_mode, "dens")
        and _raises(ValueError, D.resolve_eval_mode, "")
        is False,  # an empty value IS the default ('auto'), not an error
        "'dynamic2' / 'dens' must raise; '' must keep meaning the default",
    )
    report.add(
        "EVAL_MODES stays canonical; the alias lives in its own table",
        tuple(D.EVAL_MODES) == ("dense", "specialized", "auto")
        and aliases.get("dynamic") == "dense",
        f"EVAL_MODES={tuple(D.EVAL_MODES)} EVAL_MODE_ALIASES={aliases}",
    )

    # -- the trace cap (`EVOGP_DENSE_MAX_TREE_LEN`) and its ONE warning ------
    # ``L = min(cap, row_len)``: the cap may never change what a SHORT row
    # traces, and the default cap must accept the populations the raise was
    # measured for (33-64 nodes) while still declining longer ones.
    previous_cap = os.environ.pop(D.MAX_TREE_LEN_ENV, None)
    try:
        resolved_default = _safe(D._resolve_max_tree_len, None)
    finally:
        if previous_cap is not None:
            os.environ[D.MAX_TREE_LEN_ENV] = previous_cap
    report.add(
        "the default trace cap is 64 (EVOGP_DENSE_MAX_TREE_LEN unset)",
        resolved_default == 64 == getattr(D, "DEFAULT_MAX_TREE_LEN", None),
        f"resolved={resolved_default!r} DEFAULT_MAX_TREE_LEN="
        f"{getattr(D, 'DEFAULT_MAX_TREE_LEN', None)!r}",
    )
    D._TRACE_CAP_WARNED = False
    short_prep, short_reason, short_warnings = _cap_attempt(20, 32)
    report.add(
        "L = min(cap, row_len): a max_tree_len=32 population still traces L=32",
        getattr(short_prep, "trace_len", None) == 32 and short_reason is None,
        f"trace_len={getattr(short_prep, 'trace_len', short_reason)!r} for a "
        "20-node tree in a 32-wide row",
    )
    wide_prep, wide_reason, wide_warnings = _cap_attempt(20, 128)
    report.add(
        "a long row no longer caps the trace at 32 (L=64)",
        getattr(wide_prep, "trace_len", None) == 64 and wide_reason is None,
        f"trace_len={getattr(wide_prep, 'trace_len', wide_reason)!r} for a 20-node "
        "tree in a 128-wide row",
    )
    at_cap = _cap_attempt(64, 128)[0]
    report.add(
        "a 64-node tree fits the default cap (the raise lands)",
        getattr(at_cap, "trace_len", None) == 64
        and getattr(at_cap, "max_tree_length", None) == 64,
        f"trace_len={getattr(at_cap, 'trace_len', None)!r} "
        f"max_tree_length={getattr(at_cap, 'max_tree_length', None)!r}",
    )
    over_prep, over_reason, over_warnings = _cap_attempt(70, 128)
    report.add(
        "a 70-node tree exceeds the default cap -> DenseUnsupported",
        over_prep is None
        and "70 nodes" in (over_reason or "")
        and "cap is 64" in (over_reason or ""),
        f"reason={str(over_reason)[:70]!r}",
    )
    report.add(
        "the over-cap decline stays SILENT on numpy (dense is never the win there)",
        not over_warnings,
        f"{len(over_warnings)} warning(s)",
    )
    D._TRACE_CAP_WARNED = False
    attempts = [_cap_attempt(70, 128, backend="iree") for _ in range(3)]
    warned = [warns for _, _, warns in attempts if warns]
    report.add(
        "the over-cap decline on a device backend warns ONCE per process",
        len(warned) == 1 and all(reason for _, reason, _ in attempts),
        f"{len(warned)} warning(s) over 3 calls (all 3 still declined)",
    )
    message = warned[0][0] if warned else ""
    report.add(
        "the warning names the longest tree, the cap, the env var and the compile "
        "consequence",
        "70 nodes" in message
        and "L=64" in message
        and D.MAX_TREE_LEN_ENV in message
        and "PER DISTINCT TREE STRUCTURE" in message,
        f"message={message[:96]!r}...",
    )
    D._TRACE_CAP_WARNED = False
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        multi_prep, multi_reason, _ = _cap_attempt(4, 8, backend="iree", output_len=2)
        malformed_values, malformed_types, malformed_sizes = cap_chain(4, 8)
        malformed_types[0, 0] = int(NType.VAR)  # VAR with a declared subtree > 1
        malformed = _safe(
            D.prepare_dense,
            malformed_values,
            malformed_types,
            malformed_sizes,
            1,
            1,
            backend="iree",
        )
        other_warnings = [
            str(item.message)
            for item in recorded
            if issubclass(item.category, RuntimeWarning)
        ]
    report.add(
        "every OTHER decline reason stays silent and keeps the once-flag unset",
        multi_prep is None
        and "output_len" in (multi_reason or "")
        and isinstance(malformed, str)
        and malformed.startswith("<DenseUnsupported")
        and not other_warnings
        and D._TRACE_CAP_WARNED is False,
        f"multi-output={str(multi_reason)[:34]!r} malformed={str(malformed)[:34]!r} "
        f"warnings={len(other_warnings)}",
    )


# ---------------------------------------------------------------------------
# stage: hot path (counted host crossings on a FORCED-CHUNKED call)
# ---------------------------------------------------------------------------


def _counted_call(
    forest: Forest,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    budget: float,
) -> Tuple[Any, "Traffic"]:
    """One forced-chunked dense SR call, with every host crossing counted."""
    set_mode("dense", backend="numpy")
    with _budget(budget):
        with Traffic() as traffic:
            loss = forest.SR_fitness(inputs, labels, True, "auto")
    return loss, traffic


def _hotpath_case(
    label: str,
    report: Report,
    results: Dict[str, Any],
    *,
    budget: float,
    plan: Any,
    loss: Any,
    traffic: "Traffic",
    pop: int,
    rows: int,
    input_len: int,
) -> None:
    """The per-config host-crossing invariants (see the module docstring)."""
    print(f"-- {label}: plan {_plan_line(plan)}")
    print(f"   traffic: {json.dumps(traffic.summary(pop), sort_keys=True)}")
    results.setdefault("hotpath", {})[label] = {
        "budget_mb": budget,
        "plan": _plan_line(plan),
        **traffic.summary(pop),
    }
    moves = traffic.conversions("inputs", "x")
    report.add(
        f"[{label}] inputs converted to host exactly ONCE",
        len(moves) == 1,
        f"torch->host conversions named inputs/x: {len(moves)} "
        f"(all conversions: {[(n, s) for _, n, s in traffic.as_numpy]})",
    )
    report.add(
        f"[{label}] labels converted to host exactly ONCE",
        len(traffic.conversions("labels")) == 1,
        f"{len(traffic.conversions('labels'))} conversions",
    )
    report.add(
        f"[{label}] the three population tensors converted to host once each",
        all(
            len(traffic.conversions(name)) == 1
            for name in (
                "batch_node_value",
                "batch_node_type",
                "batch_subtree_size",
            )
        ),
        f"{len(traffic.as_numpy)} torch->host conversions in total",
    )

    readbacks = traffic.readbacks
    elements = [int(np.prod(shape)) for _, shape in readbacks]
    report.add(
        f"[{label}] every readback is (P,)-scale (no full prediction)",
        bool(elements) and max(elements) <= pop,
        f"readback shapes={[list(shape) for _, shape in readbacks]} pop_size={pop} "
        f"(a full prediction would be ({plan.p_chunk}, {plan.n_chunk}))",
    )
    report.add(
        f"[{label}] one readback per chunk",
        len(readbacks) == plan.p_chunks * plan.n_chunks,
        f"{len(readbacks)} readbacks for {plan.p_chunks}x{plan.n_chunks} chunks",
    )
    report.add(
        f"[{label}] one compiled-graph invocation per chunk",
        len(traffic.runs) == plan.p_chunks * plan.n_chunks,
        f"{len(traffic.runs)} etl.run calls for "
        f"{plan.p_chunks}x{plan.n_chunks} chunks",
    )
    creations = [int(np.prod(shape)) for _, shape in traffic.torch_creations]
    report.add(
        f"[{label}] no host->torch tensor larger than pop_size is created",
        max(creations, default=0) <= pop,
        f"torch.from_numpy shapes="
        f"{[list(shape) for _, shape in traffic.torch_creations]}",
    )
    report.add(
        f"[{label}] no labels round trip through a torch tensor",
        all(int(np.prod(shape)) != rows for _, shape in traffic.torch_creations),
        f"host->torch shapes={[list(shape) for _, shape in traffic.torch_creations]} "
        "(a labels-shaped creation means D2H -> torch -> H2D)",
    )

    # Payload PREPARATION multiplicity: backend-independent (numpy never uploads,
    # so this is the observable proxy for the device-backend upload count).
    x_preps = traffic.preparations(input_len)
    label_preps = traffic.preparations(1)
    report.add(
        f"[{label}] x prepared ONCE per n-chunk (not once per (p0, n0) chunk)",
        len(x_preps) == plan.n_chunks,
        f"{len(x_preps)} preparations of a (*, {input_len}) payload for "
        f"n_chunks={plan.n_chunks} x p_chunks={plan.p_chunks}",
    )
    report.add(
        f"[{label}] labels prepared ONCE per n-chunk",
        len(label_preps) == plan.n_chunks,
        f"{len(label_preps)} preparations of a (*, 1) payload for "
        f"n_chunks={plan.n_chunks}",
    )

    counts = traffic.upload_counts()
    if counts:  # device backend: assert the upload multiplicity directly
        x_shape = (plan.n_chunk, input_len)
        label_shape = (plan.n_chunk, 1)
        column_shape = (plan.trace_len, plan.p_chunk)
        report.add(
            f"[{label}] x uploaded ONCE per n-chunk",
            counts.get(x_shape, 0) == plan.n_chunks,
            f"shape {x_shape}: {counts.get(x_shape, 0)} uploads, expected "
            f"{plan.n_chunks}",
        )
        report.add(
            f"[{label}] labels uploaded ONCE per n-chunk",
            counts.get(label_shape, 0) == plan.n_chunks,
            f"shape {label_shape}: {counts.get(label_shape, 0)} uploads, expected "
            f"{plan.n_chunks}",
        )
        report.add(
            f"[{label}] the node columns uploaded ONCE per p-chunk each",
            counts.get(column_shape, 0) == plan.p_chunks * 7,
            f"shape {column_shape}: {counts.get(column_shape, 0)} uploads, expected "
            f"{plan.p_chunks * 7} (7 columns x p_chunks)",
        )
    else:
        report.add(
            f"[{label}] (numpy backend: no uploads to count; preparation counts above)",
            True,
            "etl.from_numpy is only used on device backends",
        )

    check_contract(f"dense SR loss contract ({label})", loss, report)
    report.add(
        f"[{label}] chunked loss shape is (pop_size,)",
        tuple(loss.shape) == (pop,),
        f"{tuple(loss.shape)}",
    )


def stage_hotpath(args: Any, report: Report, results: Dict[str, Any]) -> None:
    print("== stage hotpath: counted host crossings, forced chunked plan ==")
    pop, rows, input_len = 8, 37, 3
    forest, inputs, labels = fixture(pop, rows, input_len=input_len, seed=3)

    # (a) chunked in BOTH dimensions: the only regime in which the pre-refactor
    #     per-(p0, n0) re-preparation / re-upload of x and the node columns shows
    #     up as a multiplicity > 1.
    budget, plan = _tuned_budget(
        forest, rows, "loss", lambda p: p.p_chunks > 1 and p.n_chunks > 1
    )
    report.add(
        "a P- AND N-chunked plan exists for the hot-path fixture",
        plan is not None,
        f"tried budgets {_CANDIDATE_BUDGETS}",
    )
    if plan is not None:
        loss, traffic = _counted_call(forest, inputs, labels, budget)
        _hotpath_case(
            "both-chunked",
            report,
            results,
            budget=budget,
            plan=plan,
            loss=loss,
            traffic=traffic,
            pop=pop,
            rows=rows,
            input_len=input_len,
        )

    # (b) row-chunked only (p_chunks == 1): here the x payload preparation
    #     multiplicity bites without any P chunking.
    budget, plan = _tuned_budget(
        forest, rows, "loss", lambda p: p.p_chunks == 1 and p.n_chunks > 1
    )
    report.add(
        "a row-chunked plan exists for the hot-path fixture",
        plan is not None,
        f"tried budgets {_CANDIDATE_BUDGETS}",
    )
    if plan is not None:
        loss, traffic = _counted_call(forest, inputs, labels, budget)
        _hotpath_case(
            "row-chunked",
            report,
            results,
            budget=budget,
            plan=plan,
            loss=loss,
            traffic=traffic,
            pop=pop,
            rows=rows,
            input_len=input_len,
        )
        set_mode("dense", backend="numpy")
        with _budget(budget):
            check_writable(
                "dense SR loss is fresh and writable (chunked)",
                loss,
                lambda: forest.SR_fitness(inputs, labels, True, "auto"),
                report,
            )

    # (c) the DEFAULT budget (2048 MiB): one unchunked graph -- the realistic
    #     regime, where a full-prediction readback is (P, N) instead of (P,).
    plan = plan_for(forest, rows, budget_mb=None, variant="loss")
    loss, traffic = _counted_call(forest, inputs, labels, DEFAULT_BUDGET_MB)
    _hotpath_case(
        "unchunked (default budget)",
        report,
        results,
        budget=DEFAULT_BUDGET_MB,
        plan=plan,
        loss=loss,
        traffic=traffic,
        pop=pop,
        rows=rows,
        input_len=input_len,
    )
    set_mode("auto", backend="numpy")


# ---------------------------------------------------------------------------
# stage: parity (public API, forced dense vs forced specialized)
# ---------------------------------------------------------------------------


def _parity_pair(
    forest: Forest,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    label: str,
    report: Report,
    *,
    use_mse: bool = True,
) -> None:
    set_mode("specialized", backend="numpy")
    reference = forest.SR_fitness(inputs, labels, use_mse, "auto")
    set_mode("dense", backend="numpy")
    got = forest.SR_fitness(inputs, labels, use_mse, "auto")
    compare(
        f"{label}: dense vs specialized (use_MSE={use_mse})", reference, got, report
    )


def stage_parity(args: Any, report: Report) -> None:
    print("== stage parity: in-graph SR loss vs the specialized engine ==")
    # (a) a plain random population, MSE + MAE, single chunk
    forest, inputs, labels = fixture(16, 64, seed=11)
    _parity_pair(forest, inputs, labels, "random P=16 N=64 single chunk", report)
    _parity_pair(
        forest, inputs, labels, "random P=16 N=64 single chunk", report, use_mse=False
    )

    # (b) a padded last chunk: N is NOT a multiple of the forced chunk size
    forest, inputs, labels = fixture(8, 37, seed=12)
    for use_mse in (True, False):
        name = "MSE" if use_mse else "MAE"
        _parity_pair(forest, inputs, labels, "padded last chunk P=8 N=37", report,
                     use_mse=use_mse)
        set_mode("dense", backend="numpy")
        with _budget(2048.0):
            single = forest.SR_fitness(inputs, labels, use_mse, "auto")
        with _budget(1.0):
            chunked = forest.SR_fitness(inputs, labels, use_mse, "auto")
        report.add(
            f"chunked result == single-chunk result ({name})",
            bool(
                torch.allclose(
                    single.cpu().float(),
                    chunked.cpu().float(),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            ),
            f"max|diff|={(single - chunked).abs().max().item():.3g}",
        )

    # (c) NaN / +-inf labels and inputs (masks must match exactly)
    forest, inputs, labels = fixture(6, 24, seed=13)
    hostile = inputs.detach().clone()
    hostile[0, 0] = float("nan")
    hostile[1, 1] = float("inf")
    hostile[2, 2] = float("-inf")
    hostile[3, :] = 0.0
    hostile_labels = labels.detach().clone()
    hostile_labels[0, 0] = float("nan")
    hostile_labels[1, 0] = float("inf")
    hostile_labels[2, 0] = float("-inf")
    for use_mse in (True, False):
        _parity_pair(
            forest, hostile, hostile_labels, "NaN/+-inf labels and inputs", report,
            use_mse=use_mse,
        )

    # (d) the pop=1 Tree path (a single tree is a population of one)
    tree = forest[0]
    set_mode("specialized", backend="numpy")
    ref_tree = tree.SR_fitness(hostile, hostile_labels, True, "auto")
    set_mode("dense", backend="numpy")
    got_tree = tree.SR_fitness(hostile, hostile_labels, True, "auto")
    compare("pop=1 Tree.SR_fitness dense vs specialized", ref_tree, got_tree, report)
    report.add(
        "pop=1 loss shape is (1,)",
        tuple(got_tree.shape) == (1,),
        f"{tuple(got_tree.shape)}",
    )

    # (e) the value path is untouched: forward / batch_forward parity
    forest, inputs, labels = fixture(16, 64, seed=14)
    x_rows = torch.randn(forest.pop_size, forest.input_len, device="cuda")
    x_batch = torch.randn(24, forest.input_len, device="cuda")
    set_mode("specialized", backend="numpy")
    ref_fwd, ref_batch = forest.forward(x_rows), forest.batch_forward(x_batch)
    set_mode("dense", backend="numpy")
    got_fwd, got_batch = forest.forward(x_rows), forest.batch_forward(x_batch)
    compare("value path: Forest.forward dense vs specialized", ref_fwd, got_fwd, report)
    compare(
        "value path: Forest.batch_forward dense vs specialized",
        ref_batch,
        got_batch,
        report,
    )
    report.add(
        "value path shapes unchanged",
        tuple(got_fwd.shape) == (forest.pop_size, 1)
        and tuple(got_batch.shape) == (forest.pop_size, 24, 1),
        f"forward={tuple(got_fwd.shape)} batch={tuple(got_batch.shape)}",
    )
    set_mode("auto", backend="numpy")


# ---------------------------------------------------------------------------
# stage: the readback-free seam
# ---------------------------------------------------------------------------


#: how many chunk payloads of each kind the seam must move per call.  The
#: payload key (``p0`` for the columns, ``n0`` for ``x``) makes the multiplicity
#: exact, not an upper bound: 7 node columns per ``p``-chunk, one ``x`` per
#: ``n``-chunk (the seam fixture is the value variant, ``per_row=False``).
def _payload_expectation(plan: Any) -> Dict[str, int]:
    return {
        "columns": 7 * plan.p_chunks,
        "x": 1 if plan.per_row else plan.n_chunks,
    }


def _nan_safe_equal(a: Any, b: Any) -> Tuple[bool, str]:
    """Bit-exact NaN-aware equality of two float32 host arrays (+ detail string).

    A numeric fixture may legitimately contain NaN/±inf nodes (``DIV`` by zero,
    ``LOG`` of a negative...), so a plain ``np.array_equal`` / ``max|diff|`` is
    NaN-poisoned: the masks are compared exactly and the difference is taken
    over the finite positions only.
    """
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        return False, f"shape {x.shape} != {y.shape}"
    nan_ok = bool(np.array_equal(np.isnan(x), np.isnan(y)))
    inf_ok = bool(
        np.array_equal(np.isposinf(x), np.isposinf(y))
        and np.array_equal(np.isneginf(x), np.isneginf(y))
    )
    finite = np.isfinite(x) & np.isfinite(y)
    max_abs = (
        float(
            np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64)).max()
        )
        if finite.any()
        else 0.0
    )
    non_bit = int((finite & (x.view(np.int32) != y.view(np.int32))).sum())
    return (
        bool(nan_ok and inf_ok and max_abs == 0.0 and non_bit == 0),
        f"nan_ok={nan_ok} inf_ok={inf_ok} non_bit_exact_finite={non_bit} "
        f"max|diff|={max_abs:.3g}",
    )


def stage_seam(args: Any, report: Report) -> None:
    print("== stage seam: readback-free chunk iteration ==")
    iterate = getattr(D, "_iter_plan_chunks", None)
    if iterate is None:
        report.add(
            "the readback-free run helper exists",
            False,
            "dense_engine._iter_plan_chunks is missing: the run machinery reads "
            "every chunk back to host inside run_plan itself, so no device-resident "
            "consumer can exist",
        )
        return
    pop, rows, input_len = 8, 37, 3
    forest, inputs, _ = fixture(pop, rows, input_len=input_len, seed=21)
    # ``--backend iree`` (with ``--gpu``) resolves through the same env the
    # engine reads, so the seam is exercised with real device payloads too; the
    # default run stays on numpy, where the payload-preparation counts stand in
    # for the upload counts (nothing is uploaded at all there).
    with _backend(args.backend, args.gpu):
        plan = plan_for(forest, rows, budget_mb=1.0, variant="value")
        x_host = np.ascontiguousarray(inputs.detach().cpu().numpy(), dtype=np.float32)
        executable = D.build_dense(plan)
        backend_name, device, _ = E._resolve_backend(None, None)
        with Traffic() as traffic:
            chunks = list(
                iterate(
                    plan, executable, x_host, backend_name=backend_name, device=device
                )
            )
        report.add(
            "the seam performs ZERO host readbacks",
            len(traffic.readbacks) == 0,
            f"{len(traffic.readbacks)} ETL->host reads issued by "
            f"{_READBACK_FRAMES} while iterating "
            f"({len(traffic.tensor_numpy)} .numpy() calls in total, the rest are "
            "the numpy interpreter's internal tensor handling)",
        )
        report.add(
            "one compiled-graph invocation per chunk",
            len(traffic.runs) == plan.p_chunks * plan.n_chunks,
            f"{len(traffic.runs)} etl.run calls for "
            f"{plan.p_chunks}x{plan.n_chunks} chunks",
        )
        # Positive control: the counters above only mean something if the same
        # filter DOES see the readbacks of a consumer that reads its results
        # back.  run_plan is that consumer.
        with Traffic() as control:
            D.run_plan(plan, executable, x_host, backend_name=backend_name, device=device)
        report.add(
            "the readback/run counters are live (run_plan is seen)",
            len(control.readbacks) == plan.p_chunks * plan.n_chunks
            and len(control.runs) == plan.p_chunks * plan.n_chunks,
            f"run_plan: {len(control.readbacks)} readbacks, "
            f"{len(control.runs)} etl.run calls for "
            f"{plan.p_chunks}x{plan.n_chunks} chunks",
        )

        expectation = _payload_expectation(plan)
        column_preps = traffic.preparations(plan.trace_len)
        x_preps = traffic.preparations(input_len)
        report.add(
            "the seam PREPARES each distinct payload once per call",
            len(column_preps) == expectation["columns"]
            and len(x_preps) == expectation["x"],
            f"preparations: columns={len(column_preps)} "
            f"(expected {expectation['columns']}), x={len(x_preps)} "
            f"(expected {expectation['x']})",
        )
        counts = traffic.upload_counts()
        expected = {
            (plan.trace_len, plan.p_chunk): expectation["columns"],
            (plan.n_chunk, input_len): expectation["x"],
        }
        if backend_name == "numpy":
            report.add(
                "the seam uploads each distinct payload once per call",
                counts == {},
                "numpy backend: _pad_rows preparation counts above; "
                "etl.from_numpy is only used on device backends",
            )
        else:
            report.add(
                "the seam uploads each distinct payload once per call",
                counts == expected,
                f"uploaded { {str(k): v for k, v in counts.items()} }, expected "
                f"{ {str(k): v for k, v in expected.items()} }",
            )
        report.add(
            "one raw (un-read) result per chunk",
            len(chunks) == plan.p_chunks * plan.n_chunks,
            f"{len(chunks)} results for {plan.p_chunks}x{plan.n_chunks} chunks",
        )
        report.add(
            "the raw results are ETL tensors (device-resident on a device backend)",
            all(isinstance(chunk[1], etl.core.Tensor) for chunk in chunks),
            f"types={sorted({type(chunk[1]).__name__ for chunk in chunks})}",
        )
        assembled = np.empty((plan.pop_size, plan.n_rows), dtype=np.float32)
        for (p0, p_chunk, n0, n_chunk), raw in chunks:
            host = np.asarray(raw.numpy(), dtype=np.float32).reshape(
                plan.p_chunk, plan.n_chunk
            )
            assembled[p0 : p0 + p_chunk, n0 : n0 + n_chunk] = host[:p_chunk, :n_chunk]
        reference = D.run_plan(
            plan, executable, x_host, backend_name=backend_name, device=device
        )
        equal, detail = _nan_safe_equal(assembled, reference)
        report.add("seam-assembled rows equal run_plan's rows", equal, detail)
    set_mode("auto", backend="numpy")


# ---------------------------------------------------------------------------
# stage: measurement (warm per-call cost + per-call traffic)
# ---------------------------------------------------------------------------


def _median_call_ms(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(statistics.median(samples))


def stage_measure(args: Any, results: Dict[str, Any]) -> None:
    print(
        "== stage measure: warm per-call cost (dense forced; medians of "
        f"{args.repeats}) =="
    )
    repeats = int(args.repeats)
    cases = _parse_cases(args.cases)
    set_mode("dense", backend=args.backend)
    if args.backend == "iree" and args.gpu is not None:
        os.environ["EVOGP_ETL_IREE_DEVICE"] = str(args.gpu)

    print(
        f"{'case':>24} {'median_ms':>10} {'readback':>9} {'uploads':>8} "
        f"{'inputs_d2h':>11} {'labels_d2h':>11} {'max_rb_el':>10}"
    )

    def case(label: str, pop: int, rows: int, call: Callable[[], Any]) -> None:
        call()  # one untimed call: build/compile + warm the cache
        median_ms = _median_call_ms(call, repeats)
        with Traffic() as traffic:
            call()
        summary = traffic.summary(pop)
        row = {"case": label, "pop": pop, "rows": rows, "median_ms": median_ms, **summary}
        results.setdefault("measure", []).append(row)
        print(
            f"{label:>24} {median_ms:>10.3f} {summary['readbacks']:>9} "
            f"{summary['uploads']:>8} {summary['host_reads']['inputs']:>11} "
            f"{summary['host_reads']['labels']:>11} "
            f"{summary['readback_max_elements']:>10}"
        )

    for pop, rows in cases:
        forest, inputs, labels = fixture(pop, rows, seed=31)
        case(
            f"SR_fitness P={pop} N={rows}",
            pop,
            rows,
            lambda forest=forest, inputs=inputs, labels=labels: forest.SR_fitness(
                inputs, labels, True, "auto"
            ),
        )

    forest, inputs, labels = fixture(16, int(args.tree_rows), seed=32)
    tree = forest[0]
    case(
        f"Tree.SR_fitness N={args.tree_rows}",
        1,
        int(args.tree_rows),
        lambda tree=tree, inputs=inputs, labels=labels: tree.SR_fitness(
            inputs, labels, True, "auto"
        ),
    )

    pop, rows = cases[0]
    forest, _, _ = fixture(pop, rows, seed=33)
    x_rows = torch.randn(pop, forest.input_len, device="cuda")
    case(
        f"Forest.forward P={pop}",
        pop,
        1,
        lambda forest=forest, x_rows=x_rows: forest.forward(x_rows),
    )
    set_mode("auto", backend="numpy")


def _parse_cases(spec: str) -> List[Tuple[int, int]]:
    cases: List[Tuple[int, int]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        pop, _, rows = item.partition("x")
        cases.append((int(pop), int(rows)))
    if not cases:
        raise ValueError("--cases must be a comma-separated list like 16x64,64x1024")
    return cases


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


_STAGES = ("policy", "hotpath", "parity", "seam", "measure")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        default="policy,hotpath,parity,seam",
        help=f"comma-separated subset of {list(_STAGES)} (default: all but measure)",
    )
    parser.add_argument(
        "--measure", action="store_true", help="shorthand for --stage measure"
    )
    parser.add_argument("--backend", default="numpy", choices=["numpy", "iree"])
    parser.add_argument(
        "--cases",
        default="16x64,16x1024,64x64,64x1024",
        help="measurement grid as POPxROWS (default 16x64,16x1024,64x64,64x1024)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--tree-rows", type=int, default=64, help="rows of the pop=1 Tree case"
    )
    parser.add_argument("--gpu", type=int, default=0, help="visible IREE device index")
    parser.add_argument("--json", default=None, help="write the results JSON here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stages = [s.strip().lower() for s in args.stage.split(",") if s.strip()]
    if args.measure:
        stages = ["measure"]
    unknown = [s for s in stages if s not in _STAGES]
    if unknown:
        print(f"unknown stage(s) {unknown}: pick from {list(_STAGES)}")
        return 2
    if os.environ.get("ETL_DISABLE_LOCATIONS", "").strip() not in ("1", "true"):
        print("[warn] ETL_DISABLE_LOCATIONS=1 is not set: tracing will be ~4x slower")
    results: Dict[str, Any] = {
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "EVOGP_ETL_BACKEND": os.environ.get("EVOGP_ETL_BACKEND"),
            "ETL_DISABLE_LOCATIONS": os.environ.get("ETL_DISABLE_LOCATIONS"),
        }
    }
    print(
        f"== device-resident check: stages={stages} backend={args.backend} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} =="
    )
    report = Report()
    try:
        for stage in stages:
            if stage == "policy":
                stage_policy(args, report)
            elif stage == "hotpath":
                stage_hotpath(args, report, results)
            elif stage == "parity":
                stage_parity(args, report)
            elif stage == "seam":
                stage_seam(args, report)
            elif stage == "measure":
                stage_measure(args, results)
    except Exception as exc:  # a hard error, not a check mismatch
        import traceback

        traceback.print_exc()
        print(f"\n== HARD ERROR: {exc!r} ==")
        return 2
    results["checks"] = report.entries
    passed = sum(1 for entry in report.entries if entry["ok"])
    print(
        f"\n== {'OK' if report.ok else 'FAILED'}: {passed}/{len(report.entries)} "
        "checks passed =="
    )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"[json] {args.json}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
