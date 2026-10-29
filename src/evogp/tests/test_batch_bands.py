"""The batch route's BAND POLICY: how many merge bands a device can actually load.

``src/evogp/tree/batch_bands.py`` answers two questions for the opt-in merged
graph route: what to ASK ``evoxc.batch_functions`` for (a flat merge, an explicit
``K``, the band-SIZE model ``K = ceil(n / EVOGP_BATCH_BAND_SIZE)`` or ``"auto"``,
i.e. evoxc's static cost model) and what an ``auto`` resolution actually produced.
The reason it exists is a HARD device limit: a banded merge emits one exported
dispatch per band and the HIP/CUDA HALs bind at most 16 tensors per dispatch, so
``K <= 15`` there (``K <= 31`` on the local-task CPU HAL, uncapped on numpy and
Vulkan) and the default size model reaches an unloadable ``K = 63`` at pop 1000.
This file is the pytest home of that policy -- it used to be pinned only by the
dev battery ``src/evogp/tree/batch_band_check.py``.  It covers:

* the policy TABLES and :meth:`BandPolicy.cap` (the binding limit, the function
  count and the per-band size floor, whichever bites first);
* the four request spellings and their precedence: ``switch_ops`` -> ``auto`` ->
  an explicit ``K`` (a ``ValueError``, never a clamp, when it can not LOAD) -> the
  band-size model (CLAMPED, with ``band_clamped`` + a human ``band_note``);
* the ``auto`` passthrough onto ``describe_plan``: ``cost_estimate`` /
  ``flat_cost_estimate`` / ``band_requested`` / ``band_requested_bands`` /
  ``band_clamped`` / ``band_note`` / ``band_policy`` / ``chunk_bands`` /
  ``chunk_band_strategies``;
* the ``auto`` clamp reaching a merge report through
  :func:`batch_bands.merge_with_spec` (the one-remerge remedy, both cost
  estimates preserved, the note naming the 16-binding cap and the remedy);
* the DEFAULTS: with no knobs the band-size model still decides, so nothing about
  the default path moved;
* ``EVOGP_BATCH_CHUNK_BANDS``: off a band request forces the single-body plan, on
  every chunk repeats the request (banded + chunked composition) with the global
  function order preserved and BITWISE identical values, and the setting is part of
  both plan memo keys (a flip re-plans / misses the content level instead of
  serving the other setting's plan).

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_knobs.py`` owns the knob resolvers and the ``ceil(n / band_size)``
  model at the ``resolve_band_count`` level; this file pins the POLICY on top of it.
* ``test_batch_chunking.py`` owns the chunk partition itself and
  ``test_batch_engine.py`` the merged graph's numeric contract; the one equivalence
  test here is only about banding COMPOSED with chunking.
* ``test_batch_stages.py`` / ``test_batch_content_key.py`` own the stage counters
  and the content level's identity semantics.

Mechanics
---------
* numpy ONLY (``backend="numpy"`` passed explicitly) for everything that merges or
  builds, so no device and no iree package is involved and the whole file runs in a
  few seconds; the policy tests never touch a backend at all (they hand a fake
  ``device`` object with only a ``kind``, which is all
  :func:`batch_bands.resolve_band_policy` reads).
* No wall-clock assertion: the checks are values, flags, report fields and cache
  counters.
* Tiny fixtures (12 / 40 rows, ``L = 32``, ``N = 5``) admitted through
  ``EVOGP_BATCH_MIN_POP=1`` (the library default of 64 is the measured break-even,
  i.e. a real decline), and chunking on them through a lowered
  ``batch_knobs.MIN_CHUNK_FUNCTIONS`` (the default 32 correctly declines to split a
  12-function population).
* The autouse fixture deletes every merge knob and brackets each test with
  ``evoxc_engine.clear_cache()``, the single in-memory invalidation point (it
  resets the plan, executable and content tables through the registered hook).

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src /tmp/rocvenv/bin/python \\
        -m pytest src/evogp/tests/test_batch_bands.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import pytest
import torch

# --------------------------------------------------------------------------- #
# shared helpers: the tree-spec DSL lives in test_equivalence.py (imported by
# path, exactly like the sibling engine-round files do)
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import test_equivalence as te  # noqa: E402  (the shared spec DSL)

from evogp.tree import batch_bands as P  # noqa: E402
from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_knobs as K  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: every case is a few millisecond numpy compile)
# --------------------------------------------------------------------------- #
#: The host backend: nothing compiles to a device, nothing is uploaded.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 32
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: A population the numpy cost model actually splits (it needs enough functions
#: for a band to be worth a dispatch); the size model bands it at any band size
#: ``<= 12`` (at the shipped default of 16 it models ``K == 1``, a flat merge).
BANDED_POP = 12
#: The population at which the default size model reaches an unloadable ``K = 63``.
LARGE_POP = 1000
#: The band-policy keys ``describe_plan`` must report (the cost model's seam).
_BAND_KEYS = (
    "bands",
    "cost_estimate",
    "flat_cost_estimate",
    "band_requested",
    "band_requested_bands",
    "band_clamped",
    "band_note",
    "band_policy",
    "chunk_bands",
    "chunk_band_strategies",
)
#: The keys of the ``band_policy`` snapshot that travels in a plan report.
_POLICY_KEYS = frozenset(
    {"backend", "device_kind", "max_bands", "min_band_size", "uncapped"}
)
#: Every knob an ambient value could change under the assertions.
_ENV_KNOBS: Tuple[str, ...] = (
    B.BAND_SIZE_ENV,
    B.BANDS_ENV,
    B.SWITCH_OPS_ENV,
    B.ALIGNMENT_ENV,
    K.COMPILE_WORKERS_ENV,
    K.CHUNK_BANDS_ENV,
)
#: Mixed-arity program shapes, cycled so every row is a DISTINCT merged function
#: (the per-index constant is what makes them distinct even when the shape repeats).
_SHAPES: Sequence[Any] = (
    lambda i: te.fn(te.SIN, te.var(0)),
    lambda i: te.fn(te.ADD, te.var(1), te.var(0)),
    lambda i: te.fn(
        te.MUL,
        te.fn(te.SUB, te.var(0), te.var(1)),
        te.fn(te.MAX, te.var(1), te.const(1.5 + i)),
    ),
    lambda i: te.fn(te.IF, te.var(0), te.const(0.25 + i), te.var(1)),
)


class _DeviceKind:
    """The only thing ``resolve_band_policy`` reads off a device: its ``kind``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind


class _StubMerge:
    """A stand-in for ``evoxc.batch_functions`` that reports the bands it got.

    The numpy cost model never asks for more than one band, so the ``auto``
    CLAMP (which needs an unloadable modelled count) is driven through this stub:
    the code path is the one ``_plan_host`` runs -- :func:`merge_with_spec` -- and
    the recorded calls prove the clamped RE-MERGE happened exactly once.
    """

    def __init__(self, chosen: int) -> None:
        self.chosen = int(chosen)
        self.calls: List[Any] = []

    def __call__(
        self, graphs: Any, axes: Any, *, switch_ops: bool, bands: Any, alignment: str
    ) -> Any:
        self.calls.append(bands)
        answer: Dict[str, Any] = {
            "bands": self.chosen if bands == "auto" else int(bands or 1),
            "band_strategy": "auto" if bands == "auto" else "explicit",
        }
        if bands == "auto":
            # The model's estimates describe the CHOICE, so they must survive the
            # clamped re-merge even though the second merge runs no model.
            answer["cost_estimate"] = 1234.0
            answer["flat_cost_estimate"] = 4321.0
        merged = type("Merged", (), {})()
        merged.batch_report = type("Report", (), {"as_dict": lambda _s=None: dict(answer)})
        return merged


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Delete every merge knob, admit small populations, bracket the tables.

    The knobs are DELETED (not pinned) so "the default is 16 / off" is what the
    tests observe on a box that exports them; ``EVOGP_BATCH_MIN_POP`` is SET to
    ``1`` because the library default (64) is a real decline for every fixture
    here.  ``evoxc_engine.clear_cache()`` resets the plan, executable and content
    tables, so no entry outlives its test.
    """
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    E.clear_cache()
    try:
        yield
    finally:
        E.clear_cache()


def _programs(count: int) -> List[Any]:
    """``count`` pairwise-DISTINCT programs (one merged function per row)."""
    return [
        te.fn(te.ADD, _SHAPES[i % len(_SHAPES)](i), te.const(0.5 + i))
        for i in range(count)
    ]


def _population(count: int = BANDED_POP) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors (a new object on every call)."""
    return te.rows_from_specs(
        _programs(count), output_len=1, pad_to=ROW_LEN, tail_value=0.0
    )


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _policy(backend: str = "iree", kind: str = "rocm") -> P.BandPolicy:
    """The policy of one backend/device-kind pair (no compile, no device)."""
    return P.resolve_band_policy(backend=backend, device=_DeviceKind(kind))


def _plan(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    **knobs: Any,
) -> Any:
    """``plan_batch`` on the shape grid (value variant: pure host work)."""
    return B.plan_batch(
        values, types, sizes, INPUT_LEN, 1, x, per_row=False, backend=BACKEND, **knobs
    )


def _info(plan: Any) -> Dict[str, Any]:
    """``describe_plan`` of ``plan`` (``stage_ms`` is measurement, dropped here)."""
    info = dict(B.describe_plan(plan))
    info.pop("stage_ms", None)
    return info


def _value(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    **knobs: Any,
) -> torch.Tensor:
    """``evaluate_batch`` on the shape grid with the numpy backend pinned."""
    return B.evaluate_batch(
        values, types, sizes, INPUT_LEN, 1, x, per_row=False, backend=BACKEND, **knobs
    )


def _lower_min_chunk_functions(monkeypatch: pytest.MonkeyPatch, value: int = 4) -> None:
    """Force chunking on the tiny fixtures.

    ``MIN_CHUNK_FUNCTIONS`` is read by ``batch_programs`` THROUGH the
    ``batch_knobs`` module object, so patching the module attribute is the
    documented seam (the default 32 correctly declines to split this fixture).
    """
    monkeypatch.setattr(K, "MIN_CHUNK_FUNCTIONS", value)


# =========================================================================== #
# 1. the policy tables and the capability arithmetic
# =========================================================================== #
def test_the_binding_tables_are_the_documented_ones() -> None:
    """The HAL tables are the policy's whole truth: 16/64 bindings, 15/31 safe.

    Reading them here spells each constant once, so a silently "relaxed" table
    (the way an unloadable ``K`` would come back) fails a test instead of changing
    a graph that still runs on numpy.
    """
    assert P.DISPATCH_BINDING_LIMITS == {
        "cuda": 16,
        "rocm": 16,
        "cpu": 64,
        "vulkan": None,
    }
    assert P.SAFE_BAND_COUNTS == {"cuda": 15, "rocm": 15, "cpu": 31, "vulkan": None}
    assert P.MIN_BAND_SIZE == {"cuda": 8, "rocm": 8, "cpu": 1, "vulkan": 1}
    assert P.BAND_REQUESTS == ("flat", "explicit", "size", "auto")


@pytest.mark.parametrize(
    "backend, kind, max_bands, min_band_size",
    [
        ("iree", "rocm", 15, 8),
        ("iree", "cuda", 15, 8),
        ("iree", "cpu", 31, 1),
        ("iree", "vulkan", None, 1),
        ("numpy", "cpu", None, 1),
    ],
    ids=["rocm", "cuda", "cpu", "vulkan", "numpy"],
)
def test_resolve_band_policy_applies_the_table_per_device_kind(
    backend: str, kind: str, max_bands: Any, min_band_size: int
) -> None:
    """Every (backend, kind) pair gets its own cap and per-band size floor.

    ``uncapped`` is what lets the interpreter and Vulkan accept any ``K``; the two
    GPU kinds are the ones the default size model can overshoot, and the CPU HAL
    sits in between.
    """
    policy = _policy(backend, kind)
    assert (policy.backend, policy.device_kind) == (backend, kind)
    assert policy.max_bands == max_bands
    assert policy.min_band_size == min_band_size
    assert policy.uncapped is (max_bands is None)
    assert policy.label == f"{backend}/{kind}"
    assert set(policy.as_dict()) == _POLICY_KEYS
    assert policy.as_dict()["uncapped"] is policy.uncapped


@pytest.mark.parametrize(
    "kind, n_functions, expected",
    [
        ("rocm", 200, 15),  # the binding limit bites
        ("rocm", 1000, 15),
        ("rocm", 10, 1),  # the per-band size floor (8) bites before the limit
        ("rocm", 8, 1),
        ("rocm", 120, 15),  # 120 // 8 = 15, floor == limit
        ("cpu", 1000, 31),  # the CPU HAL allows 31
        ("cpu", 10, 10),  # ... and its floor is 1, so small counts pass through
    ],
    ids=["rocm_200", "rocm_1000", "rocm_10", "rocm_8", "rocm_120", "cpu_1000", "cpu_10"],
)
def test_the_cap_is_the_binding_limit_count_and_band_floor(
    kind: str, n_functions: int, expected: int
) -> None:
    """``cap(n) = max(1, min(max_bands, n, max(1, n // min_band_size)))``.

    Whichever of the three bites first wins, and the answer is always ``>= 1`` so a
    tiny population still merges as one band instead of degenerating to zero.
    """
    policy = _policy("iree", kind)
    assert policy.cap(n_functions) == expected
    assert 1 <= policy.cap(n_functions) <= n_functions
    assert policy.cap(3) <= 3, "the cap can never exceed the function count"


def test_an_uncapped_policy_caps_at_the_function_count() -> None:
    """numpy/Vulkan still clamp ``K`` into ``1 <= K <= n`` (the caller's job too)."""
    for backend, kind in (("numpy", "cpu"), ("iree", "vulkan")):
        policy = _policy(backend, kind)
        assert policy.cap(200) == 200
        assert policy.cap(1) == 1
        assert "no buffer-binding cap" in policy.reason or (
            "no exported-dispatch binding limit" in policy.reason
        ), policy.reason


# =========================================================================== #
# 2. the request spellings: auto, explicit K, the size model
# =========================================================================== #
def test_the_auto_spelling_resolves_to_a_request_without_a_count() -> None:
    """``bands="auto"`` cannot know its ``K`` before the merge, so it has none.

    ``evoxc_bands`` is what travels to ``batch_functions``: the string ``"auto"``,
    never a number the policy would have had to invent.
    """
    spec = P.plan_band_spec(200, bands=K.BANDS_AUTO, policy=_policy())
    assert spec.requested == "auto"
    assert spec.num_bands is None
    assert spec.evoxc_bands == "auto"
    assert (spec.clamped, spec.note) == (False, None)
    assert spec.as_dict() == {
        "requested": "auto",
        "num_bands": None,
        "clamped": False,
        "note": None,
    }


def test_switch_ops_still_forces_the_flat_merge_over_any_band_request() -> None:
    """Banding is never combined with op-switching: that precedence is untouched."""
    spec = P.plan_band_spec(200, bands=K.BANDS_AUTO, switch_ops=True, policy=_policy())
    assert (spec.requested, spec.num_bands) == ("flat", 1)
    assert spec.evoxc_bands is None, "a flat merge must travel as None, not K == 1"


def test_an_explicit_k_within_the_cap_is_accepted_unchanged() -> None:
    """A loadable explicit ``K`` is used as asked -- a policy never second-guesses it."""
    spec = P.plan_band_spec(200, bands=15, policy=_policy())
    assert (spec.requested, spec.num_bands) == ("explicit", 15)
    assert spec.as_dict()["clamped"] is False
    assert spec.evoxc_bands == 15
    flat = P.plan_band_spec(200, bands=1, policy=_policy())
    assert (flat.requested, flat.num_bands, flat.evoxc_bands) == ("flat", 1, None)


def test_the_auto_clamp_lowers_an_unloadable_modelled_count() -> None:
    """An ``auto`` answer over the cap is clamped, and the note says why + remedy.

    The note is the whole deliverable of the clamp: which chooser picked the count,
    the pre-clamp count, the 16-binding cap with its runtime source, the clamped
    count, and what the operator can do about it.
    """
    capped, note = P.clamp_auto_band_count(40, 200, _policy())
    assert capped == 15
    assert note is not None
    assert "auto (the evoxc cost model) chose 40 bands" in note
    assert "16 tensors" in note
    assert "K <= 15" in note
    assert "ROCm/HIP" in note
    assert "the merge was clamped to 15 bands" in note
    assert f"{K.BANDS_ENV}=K with K <= 15" in note


def test_a_compliant_auto_count_is_left_alone_with_no_note() -> None:
    """No clamp means no note and no re-merge -- the quiet path must stay quiet."""
    assert P.clamp_auto_band_count(3, 200, _policy()) == (3, None)
    assert P.clamp_auto_band_count(1, 200, _policy()) == (1, None)


def test_the_auto_clamp_reaches_the_merge_report_once() -> None:
    """``merge_with_spec`` stamps ``band_clamped``/``band_note`` and re-merges once.

    The device never gets to SEE an unloadable count: the first merge reports what
    the model chose, the clamped count is what is actually asked for the second
    time, and both cost estimates survive onto the final report (they describe the
    choice, which only the first merge ran).
    """
    note = P.clamp_auto_band_count(40, 200, _policy())[1]
    merge = _StubMerge(chosen=40)
    _merged, report = P.merge_with_spec(
        merge,
        [],
        0,
        P.BandSpec("auto", None, False, None, _policy()),
        switch_ops=False,
        alignment="variant",
        auto_total=200,
    )
    assert merge.calls == ["auto", 15], "the second merge must ask for the clamped K"
    assert report["bands"] == 15
    assert report["band_clamped"] is True
    assert report["band_note"] == note
    assert report["band_requested"] == "auto"
    assert report["band_requested_bands"] == 40, "the PRE-clamp auto K is kept"
    assert report["band_policy"]["device_kind"] == "rocm"
    assert (report["cost_estimate"], report["flat_cost_estimate"]) == (1234.0, 4321.0)


def test_an_explicit_unloadable_k_raises_naming_the_cap_and_the_remedy() -> None:
    """An explicit ``K`` a GPU cannot LOAD is a configuration bug, not a wish.

    Clamping it silently would build a graph that fails at LOAD time; the message
    has to name the variable, the pre-clamp value, the real limit, the HAL that
    enforces it and the two remedies (lower ``K`` / raise the band size to >= 8).
    """
    with pytest.raises(ValueError) as excinfo:
        P.plan_band_spec(200, bands=48, policy=_policy())
    text = str(excinfo.value)
    assert f"{K.BANDS_ENV}=48" in text
    assert "iree/rocm" in text
    assert "K <= 15" in text
    assert "16 tensors" in text
    assert "ROCm/HIP" in text
    assert "native_executable.c" in text
    assert "Lower it" in text
    assert f"raise {K.BAND_SIZE_ENV} to 8 or more" in text
    assert "K <= 31" in text, "the message states the CPU limit for contrast too"


def test_an_explicit_unloadable_k_raises_naming_the_cpu_cap() -> None:
    """The CPU HAL allows more, so its refusal (and remedy) carries its own numbers."""
    with pytest.raises(ValueError) as excinfo:
        P.plan_band_spec(200, bands=48, policy=_policy("iree", "cpu"))
    text = str(excinfo.value)
    assert "K <= 31" in text
    assert "local-task CPU" in text
    assert f"raise {K.BAND_SIZE_ENV} to 1 or more" in text
    assert "K <= 15" in text, "the message states the CUDA/ROCm limit for contrast too"


@pytest.mark.parametrize(
    "backend, kind", [("numpy", "cpu"), ("iree", "vulkan")], ids=["numpy", "vulkan"]
)
def test_uncapped_policies_accept_any_explicit_k(backend: str, kind: str) -> None:
    """No binding limit means no refusal: the interpreter and Vulkan take ``K = 48``."""
    spec = P.plan_band_spec(200, bands=48, policy=_policy(backend, kind))
    assert (spec.requested, spec.num_bands) == ("explicit", 48)
    assert spec.evoxc_bands == 48


def test_the_out_of_range_refusal_still_names_the_variable() -> None:
    """``K > n_functions`` keeps round 1's message (the policy never sees it)."""
    with pytest.raises(ValueError) as excinfo:
        P.plan_band_spec(4, bands=5, policy=_policy())
    assert f"{K.BANDS_ENV}=5" in str(excinfo.value)
    assert "outside 1..4" in str(excinfo.value)
    with pytest.raises(ValueError):
        P.plan_band_spec(0, bands=1)


def test_resolve_band_count_refuses_the_auto_spelling_with_a_value_error() -> None:
    """The count-only helper cannot answer ``auto``: it must say so, not guess.

    A ``TypeError`` here would be the regression (``"auto"`` silently reaching an
    ``int()`` somewhere), so the exception TYPE is part of the contract.
    """
    with pytest.raises(ValueError) as excinfo:
        K.resolve_band_count(BANDED_POP, bands=K.BANDS_AUTO)
    text = str(excinfo.value)
    assert K.BANDS_ENV in text
    assert f"'{K.BANDS_AUTO}'" in text
    assert "cost model at merge time" in text
    assert "no band count to resolve here" in text


# =========================================================================== #
# 3. the size model's clamp (the reason this policy exists)
# =========================================================================== #
def test_the_band_size_model_is_clamped_at_a_large_population() -> None:
    """Pop 1000 with the default band size models ``K = 63``, which no GPU can load.

    This is the case the policy was written for: ``ceil(1000 / 16) == 63`` bands,
    15 loadable, so the spec is clamped and carries the note -- instead of the
    generation dying inside IREE's LOAD step.
    """
    default_band_size = K.resolve_band_size()
    assert default_band_size == 16
    spec = P.plan_band_spec(LARGE_POP, policy=_policy())
    assert spec.requested == "size"
    assert spec.num_bands == 15
    assert spec.clamped is True
    assert spec.evoxc_bands == 15
    assert spec.note is not None
    assert f"{K.BAND_SIZE_ENV}=16" in spec.note
    assert "chose 63 bands" in spec.note
    assert "can only load 15" in spec.note
    assert f"explicit {K.BANDS_ENV}=K with K <= 15" in spec.note


def test_a_size_model_count_within_the_cap_is_not_clamped() -> None:
    """A small population's modelled ``K`` passes through with no note at all."""
    spec = P.plan_band_spec(40, policy=_policy())
    assert (spec.requested, spec.num_bands) == ("size", 3)
    assert (spec.clamped, spec.note) == (False, None)
    assert spec.evoxc_bands == 3


# =========================================================================== #
# 4. what a real plan reports (the describe_plan seam)
# =========================================================================== #
def test_no_knob_plan_keeps_the_band_size_model() -> None:
    """The default path is unchanged: ``requested == "size"``, ``K = ceil(n/16)``.

    Round 2 added ``auto`` and the clamp; it must not have moved the default, so a
    caller who sets nothing still gets the flat-ish ``explicit`` size model and no
    cost-model claim in the report.
    """
    info = _info(_plan(*_population(40), _x()))
    expected = -(-info["n_functions"] // K.DEFAULT_BAND_SIZE)
    assert info["n_functions"] == 40
    assert expected == 3, "the fixture must be large enough to band at size 16"
    assert info["band_requested"] == "size"
    assert info["bands"] == info["band_count"] == expected
    assert info["band_strategy"] == "explicit"
    assert len(info["band_sizes"]) == expected
    assert info["band_clamped"] is False
    assert info["band_note"] is None
    assert info["cost_estimate"] is None, "the size model must not claim a model run"
    assert info["band_policy"]["uncapped"] is True


def test_an_auto_plan_reports_the_cost_model_choice() -> None:
    """``EVOGP_BATCH_BANDS=auto`` on numpy: the model decides and the plan says so.

    numpy is uncapped, so nothing is clamped here -- the plan records the request,
    both estimates and the policy, i.e. everything a runner needs to explain the
    merge it just built.
    """
    info = _info(_plan(*_population(BANDED_POP), _x(), bands=K.BANDS_AUTO))
    assert info["band_strategy"] == "auto"
    assert info["band_requested"] == "auto"
    assert info["cost_estimate"] is not None
    assert info["flat_cost_estimate"] is not None
    assert info["band_requested_bands"] is not None
    assert info["band_clamped"] is False
    assert info["band_note"] is None
    assert info["band_policy"]["uncapped"] is True
    assert info["band_policy"]["device_kind"] == "cpu"
    assert info["bands"] == info["band_count"] == 1
    assert info["chunk_bands"] == [1]
    assert info["chunk_band_strategies"] == ["auto"]


def test_describe_plan_reports_every_band_policy_key() -> None:
    """The report's band keys are a contract: every one of them is always present.

    The values differ per plan shape (a chunked plan reports no single
    ``band_sizes``), so this test only pins PRESENCE and TYPE -- a key that
    silently disappears would break every runner reading the band story.
    """
    info = _info(_plan(*_population(BANDED_POP), _x()))
    missing = [key for key in _BAND_KEYS if key not in info]
    assert not missing, f"describe_plan lost band keys: {missing}"
    assert isinstance(info["bands"], int)
    assert isinstance(info["band_clamped"], bool)
    assert isinstance(info["band_policy"], dict)
    assert set(info["band_policy"]) == _POLICY_KEYS
    assert isinstance(info["chunk_bands"], list)
    assert all(isinstance(count, int) for count in info["chunk_bands"])
    assert isinstance(info["chunk_band_strategies"], list)
    assert len(info["chunk_band_strategies"]) == len(info["chunk_bands"])


# =========================================================================== #
# 5. EVOGP_BATCH_CHUNK_BANDS: banding composed with the chunked compile
# =========================================================================== #
def test_the_chunk_bands_knob_defaults_off_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off by default; the explicit argument wins; a malformed value names the var."""
    assert K.DEFAULT_CHUNK_BANDS is False
    assert B.resolve_chunk_bands() is False
    assert B.resolve_chunk_bands(True) is True
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "1")
    assert B.resolve_chunk_bands() is True
    assert B.resolve_chunk_bands(False) is False, "the explicit argument must win"
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "0")
    assert B.resolve_chunk_bands() is False
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "sometimes")
    with pytest.raises(ValueError) as excinfo:
        B.resolve_chunk_bands()
    assert K.CHUNK_BANDS_ENV in str(excinfo.value)


def test_a_band_request_forces_the_single_body_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the composition knob, ``auto`` + 4 workers still plans ONE body.

    The round-1 band sweep was measured on the single-body path, so a band request
    must keep meaning exactly that -- a silent chunked plan would change which graph
    those numbers came from.
    """
    _lower_min_chunk_functions(monkeypatch)
    info = _info(
        _plan(
            *_population(BANDED_POP),
            _x(),
            bands=K.BANDS_AUTO,
            chunk_bands=False,
            compile_workers=4,
        )
    )
    assert info["compile_workers"] == 1
    assert info["chunk_count"] == 1
    assert info["chunk_band_strategies"] == ["auto"]
    assert info["band_requested"] == "auto"


def test_chunk_bands_composes_the_request_with_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With it on, EACH chunk repeats the request sized to that chunk.

    The chunk geometry itself is ``test_batch_chunking.py``'s business; what is
    pinned here is that the band request survives the split (one strategy per
    chunk) and that the global function order/count is untouched, which is what
    keeps the assembled values identical.
    """
    _lower_min_chunk_functions(monkeypatch)
    x = _x()
    off = _info(
        _plan(*_population(BANDED_POP), x, bands=K.BANDS_AUTO, chunk_bands=False,
              compile_workers=4)
    )
    on = _info(
        _plan(*_population(BANDED_POP), x, bands=K.BANDS_AUTO, chunk_bands=True,
              compile_workers=4)
    )
    assert on["compile_workers"] == on["chunk_count"] == 3
    assert on["chunk_sizes"] == [4, 4, 4]
    assert on["chunk_band_strategies"] == ["auto"] * 3
    assert len(on["chunk_bands"]) == 3
    assert on["n_functions"] == off["n_functions"] == BANDED_POP
    assert sum(on["chunk_sizes"]) == BANDED_POP
    assert on["band_requested"] == "auto"


def test_chunked_banded_values_are_bitwise_the_single_body_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition is a COMPILATION strategy: the numbers may not move at all.

    On the numpy backend the two plans must agree bit for bit (``torch.equal``,
    never a tolerance), because a chunk owns its rows and keeps its global offset.
    The RUN entry points take no ``chunk_bands`` argument (only ``plan_batch``
    does), so the composition is steered the way a caller steers it: through
    ``EVOGP_BATCH_CHUNK_BANDS``.
    """
    _lower_min_chunk_functions(monkeypatch)
    x = _x()
    values, types, sizes = _population(BANDED_POP)
    monkeypatch.delenv(K.CHUNK_BANDS_ENV, raising=False)
    single = _value(values, types, sizes, x, bands=K.BANDS_AUTO, compile_workers=4)
    E.clear_cache()
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "1")
    chunked = _value(values, types, sizes, x, bands=K.BANDS_AUTO, compile_workers=4)
    assert chunked.shape == single.shape == (BANDED_POP, N_ROWS, 1)
    assert torch.equal(chunked, single), (
        f"chunked+banded values drifted: max|diff|="
        f"{float((chunked - single).abs().max()):.3g}"
    )
    assert torch.isfinite(chunked).all()


def test_flipping_chunk_bands_replans_instead_of_serving_the_other_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The setting is part of the plan memo KEY: a flip must build a new plan.

    Both settings are legal for the same population, so a key that forgot
    ``chunk_bands`` would serve a chunked plan to an un-chunked call (or the other
    way round) -- a wrong graph with the right numbers.  The plan counter is what
    makes that invisible-otherwise failure observable.
    """
    _lower_min_chunk_functions(monkeypatch)
    values, types, sizes = _population(BANDED_POP)
    x = _x()
    plans: List[Any] = []
    original = B._plan_host

    def recording(*args: Any, **kwargs: Any) -> Any:
        plan = original(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(B, "_plan_host", recording)
    try:
        off_values = _value(
            values, types, sizes, x, bands=K.BANDS_AUTO, compile_workers=4
        )
        assert len(plans) == 1, "the first call must plan"
        assert _info(plans[0])["chunk_bands"] == [1]
        monkeypatch.setenv(K.CHUNK_BANDS_ENV, "1")
        on_values = _value(values, types, sizes, x, bands=K.BANDS_AUTO, compile_workers=4)
        assert len(plans) == 2, "the flipped setting was served the off plan"
        assert _info(plans[1])["chunk_bands"] == [1, 1, 1]
        assert _info(plans[1])["compile_workers"] == 3
        monkeypatch.setenv(K.CHUNK_BANDS_ENV, "0")
        back_values = _value(
            values, types, sizes, x, bands=K.BANDS_AUTO, compile_workers=4
        )
        assert len(plans) == 2, "flipping back must be served the original off plan"
    finally:
        monkeypatch.setattr(B, "_plan_host", original)
    assert torch.equal(on_values, off_values)
    assert torch.equal(back_values, off_values)


def test_chunk_bands_is_part_of_the_content_key_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REBUILT-equal population at the other setting must not hit the content level.

    The content table serves a population that is only equal in CONTENT, so its key
    must carry every plan parameter as well -- ``chunk_bands`` included.  A served
    entry would hand back the other setting's plan (and count a hit).
    """
    _lower_min_chunk_functions(monkeypatch)
    x = _x()
    E.clear_cache()
    monkeypatch.delenv(K.CHUNK_BANDS_ENV, raising=False)
    _value(*_population(BANDED_POP), x, bands=K.BANDS_AUTO)
    first = B.batch_content_stats()
    assert (first["batch_content_hits"], first["batch_content_misses"]) == (0, 1)
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "1")
    _value(*_population(BANDED_POP), x, bands=K.BANDS_AUTO)
    second = B.batch_content_stats()
    assert second["batch_content_hits"] == 0, (
        "the other chunk_bands setting was served a content hit"
    )
    assert second["batch_content_misses"] == 2
    monkeypatch.setenv(K.CHUNK_BANDS_ENV, "0")
    _value(*_population(BANDED_POP), x, bands=K.BANDS_AUTO)
    third = B.batch_content_stats()
    assert third["batch_content_hits"] == 1, "the matching setting must be served"
    assert third["batch_content_misses"] == 2
