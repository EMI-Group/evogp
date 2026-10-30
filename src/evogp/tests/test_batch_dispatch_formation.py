"""The batch route's ``dispatch_formation`` compile option: knob, policy, seam.

``EVOGP_BATCH_DISPATCH_FORMATION`` steers ONE iree compile option of the merged
batch graph (``--iree-dispatch-creation-enable-aggressive-fusion``, 596 -> 548
IREE dispatches and -9 % warm per call at pop 1000 / rows 1024, compile wall
unchanged, identical numerics).  It is a knob whose whole value is invisible to a
numeric test -- ``default`` and ``aggressive`` produce the same numbers -- so the
thing that HAS to be pinned is the plumbing: the spelling the env asks for, the
spelling the compile actually receives, and the identity of the compiled body in
between (a re-spelled option is a different memo entry, never a hit).

This file pins, in that order:

* **resolution** -- ``resolve_dispatch_formation``: ``auto`` by default, the
  case/whitespace-folded env spellings (``off`` included, folded onto ``default``),
  the explicit-argument-wins rule, the per-call environment read, and a
  ``ValueError`` naming the variable for every unusable value (``junk``, a bool,
  and the ``none`` / ``flat`` spellings of the OTHER ``EVOGP_BATCH_*`` knobs, which
  mean nothing here);
* **the policy table** -- ``effective_dispatch_formation(spelling, backend=, gpu=,
  supported=)``: ``aggressive`` only for ``iree`` + a GPU target + an installed
  evoxc that knows the option, every explicit degradation (a non-iree backend, a
  CPU target, an older evoxc) and the ``default``/``off`` request that never derives
  a flag at all;
* **the plan's own answer** -- ``plan_dispatch_formation(backend, device)`` reads
  the target off the SAME band policy the merge uses (``rocm``/``cuda`` -> a GPU
  target, ``cpu`` -> not), consults the evoxc feature probe only for ``iree`` and
  records BOTH spellings (``dispatch_formation`` = effective,
  ``dispatch_formation_requested`` = the request) on the ``BatchPlan`` and in
  ``describe_plan()`` without disturbing a single pre-existing report key;
* **the probe** -- ``supports_dispatch_formation``: ``False`` for ``numpy``,
  answered off the adapter's OWN ``KNOWN_OPTIONS["compile"]`` table, ``False`` for
  ``None`` / an empty set / a set without the option, ``True`` only when the option
  is declared, never raising when the walk throws, memoized per backend with
  ``refresh=True`` re-probing, and -- the reason it walks the adapter MODULE --
  never importing ``iree`` itself;
* **the compile seam** -- what ``batch_chunks.build_bodies`` receives:
  ``dispatch_formation="aggressive"`` for an aggressive plan (single body AND
  chunked), NO key at all for a ``default`` plan (byte-identical to the pre-knob
  build), and a caller's options dict that is never mutated;
* **identity** -- the spelling is in the plan memo key AND in
  ``batch_keys._executable_key``'s frozen options, so two spellings are two plans
  and two compiled bodies, and only a repeat of the SAME spelling hits.

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_knobs.py`` owns the other ``EVOGP_BATCH_*`` resolvers and the
  eight-key ``batch_cache_stats()`` surface; this file only counts its own deltas.
* ``test_batch_chunking.py`` owns the chunk geometry and its numerics; here a
  chunked plan is used only as a second place the option must arrive.
* ``test_batch_engine.py`` / ``test_batch_route.py`` own the numeric contract of
  the merged graph, which this option never changes.
* ``test_batch_artifacts.py`` owns the disk tier that keys on the same options.

Mechanics
---------
* numpy ONLY (``backend="numpy"`` passed explicitly), CPU tensors, no device, no
  iree compile, no wall-clock assertion -- the checks are strings, plan records,
  captured option dicts and cache counters.
* Tiny shapes (four distinct programs, ``N = 5`` rows, ``L = 16``) admitted through
  ``EVOGP_BATCH_MIN_POP=1`` -- the library default of 64 is the measured
  break-even, i.e. a real decline.
* The installed evoxc is ASKED, never assumed: a test that needs the option to
  exist skips on a checkout that predates it (``_require_option``), while every
  policy test drives the probe itself through the documented seam
  (``batch_chunks._adapter_compile_options`` / ``batch_programs.batch_chunks``), so
  the policy table is pinned identically on both checkouts.
* ``batch_chunks._SUPPORT_CACHE`` is process-global and memoizes the probe, so the
  autouse fixture replaces it with a fresh dict per test and
  ``evoxc_engine.clear_cache()`` brackets the front tables.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_dispatch_formation.py -q \\
        -p no:cacheprovider
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

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

from evogp.tree import batch_build  # noqa: E402
from evogp.tree import batch_chunks  # noqa: E402
from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_keys  # noqa: E402
from evogp.tree import batch_knobs  # noqa: E402
from evogp.tree import batch_programs  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: this file is about one compile option)
# --------------------------------------------------------------------------- #
#: The host backend: nothing is uploaded, nothing is compiled to a device.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: FOUR distinct programs -- three computed ones and one fully constant tree.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
POP_SIZE = len(_PROGRAMS)
#: The knob this file owns, plus every OTHER merge knob (which could otherwise move
#: the plan shape out from under an assertion).
_ENV_KNOBS: Tuple[str, ...] = (
    batch_knobs.DISPATCH_FORMATION_ENV,
    batch_knobs.BAND_SIZE_ENV,
    batch_knobs.BANDS_ENV,
    batch_knobs.SWITCH_OPS_ENV,
    batch_knobs.ALIGNMENT_ENV,
    batch_knobs.COMPILE_WORKERS_ENV,
)
#: Every ``describe_plan`` key the route documents, the two compile-option keys
#: included: the report is a public seam, so the set is pinned whole (an added,
#: renamed or dropped key is a change to what every record reads).
_DESCRIBE_KEYS = frozenset(
    {
        "variant",
        "per_row",
        "use_mse",
        "pop_size",
        "functions",
        "const_functions",
        "row_index_collisions",
        "input_len",
        "output_len",
        "n_rows",
        "band_count",
        "band_sizes",
        "band_strategy",
        "switch_ops",
        "alignment",
        "emitted_ops",
        "value_classes",
        "shared_classes",
        "stacked_constants",
        "output_muxes",
        "n_functions",
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
        "dispatch_formation",
        "dispatch_formation_requested",
        "compile_workers",
        "chunk_count",
        "chunk_sizes",
        "stage_ms",
        "specs",
        "spec_dtypes",
        "backend",
        "device",
    }
)


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the knobs, admit small populations, reset the memo and the probe.

    The knobs are DELETED (not pinned to a default) so "the default is ``auto``"
    means the same thing on a box that exports them; ``EVOGP_BATCH_MIN_POP`` is SET
    to ``1`` because the library default (64) is a real decline for every fixture
    here.  ``_SUPPORT_CACHE`` is the probe's process-global memo: a fresh dict per
    test keeps an earlier test's answer (or a monkeypatched option table) from
    deciding this one's policy, and ``clear_cache()`` brackets the front tables.
    """
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    monkeypatch.setattr(batch_chunks, "_SUPPORT_CACHE", {})
    E.clear_cache()
    try:
        yield
    finally:
        E.clear_cache()


def _population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors of ``_PROGRAMS``."""
    return te.rows_from_specs(_PROGRAMS, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int = N_ROWS, seed: int = 101) -> torch.Tensor:
    """Deterministic ``(rows, 1)`` SR labels from a LOCAL generator."""
    return te.x_random(rows, 1, seed=seed)


def _plan(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    **knobs: Any,
) -> Any:
    """``plan_batch`` with the shape grid, the numpy backend and the loss variant.

    ``plan_batch`` is pure host work (decode, dedup, trace, merge), so the plan
    records cost no compile -- which is what lets a test inspect the resolved
    spelling without building anything.
    """
    return B.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=False,
        variant=B.LOSS_VARIANT,
        labels=_labels(x.shape[0]),
        backend=BACKEND,
        **knobs,
    )


def _require_option() -> None:
    """Skip when the installed evoxc does not declare the compile option.

    The feature is a pure optimisation and the route degrades silently on an evoxc
    that predates it, so a test that wants to OBSERVE ``aggressive`` needs the
    option to exist.  Everything else in this file drives the probe itself.
    """
    if not batch_chunks.supports_dispatch_formation("iree"):
        pytest.skip(
            "the installed evoxc does not declare the dispatch_formation compile "
            "option (the route degrades to `default` -- see the knob docs)"
        )


class _OptionsCapture:
    """Wrap ``batch_chunks.build_bodies`` and record what each build received."""

    def __init__(self) -> None:
        self.calls: list = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = batch_chunks.build_bodies

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls.append({"args": args, "kwargs": dict(kwargs)})
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", "build_bodies")
        monkeypatch.setattr(batch_chunks, "build_bodies", wrapper)

    def options(self) -> Dict[str, Any]:
        """The options dict of the LAST build (exactly one call per assertion)."""
        assert len(self.calls) == 1, f"expected one build, got {len(self.calls)}"
        return self.calls[-1]["kwargs"]["options"]

    def merges(self) -> list:
        """The merged graphs of the LAST build (one per compiled body)."""
        assert len(self.calls) == 1, f"expected one build, got {len(self.calls)}"
        return list(self.calls[-1]["args"][0])


class _Gate:
    """A controllable ``plan_dispatch_formation`` stand-in (the identity tests)."""

    def __init__(self) -> None:
        self.effective = batch_knobs.DISPATCH_FORMATION_DEFAULT
        self.calls = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> Tuple[str, str]:
        self.calls += 1
        return batch_knobs.DISPATCH_FORMATION_AUTO, self.effective

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch BOTH copies: the facade keys on it, ``batch_programs`` records it."""
        monkeypatch.setattr(batch_programs, "plan_dispatch_formation", self)
        monkeypatch.setattr(B, "plan_dispatch_formation", self)


# =========================================================================== #
# 1. resolution: the knob and its spellings
# =========================================================================== #
def test_the_knob_constants_are_the_documented_ones() -> None:
    """The names and spellings the knob documents (and the fixture clears)."""
    assert batch_knobs.DISPATCH_FORMATION_ENV == "EVOGP_BATCH_DISPATCH_FORMATION"
    assert batch_knobs.DISPATCH_FORMATION_ENV == B.DISPATCH_FORMATION_ENV
    assert batch_knobs.DEFAULT_DISPATCH_FORMATION == "auto"
    assert batch_knobs.DISPATCH_FORMATIONS == ("auto", "aggressive", "default")
    assert B.DISPATCH_FORMATIONS == batch_knobs.DISPATCH_FORMATIONS
    assert (
        batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
        == "aggressive"
        and batch_knobs.DISPATCH_FORMATION_DEFAULT == "default"
    )
    # The two effective spellings are what a compile can receive; `auto` is a
    # request for the policy, never an effective value.
    assert "auto" not in (
        batch_knobs.DISPATCH_FORMATION_AGGRESSIVE,
        batch_knobs.DISPATCH_FORMATION_DEFAULT,
    )


def test_the_default_is_auto() -> None:
    """Unset/empty means ``auto`` -- the target policy decides, not the caller."""
    assert B.resolve_dispatch_formation() == "auto"
    assert batch_knobs.resolve_dispatch_formation() == "auto"
    assert batch_knobs.resolve_dispatch_formation("") == "auto"
    assert batch_knobs.resolve_dispatch_formation("   ") == "auto"


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param("auto", "auto", id="auto"),
        pytest.param("AUTO", "auto", id="auto-upper"),
        pytest.param("aggressive", "aggressive", id="aggressive"),
        pytest.param("AgGrEsSiVe", "aggressive", id="aggressive-mixed"),
        pytest.param("  aggressive  ", "aggressive", id="aggressive-padded"),
        pytest.param("default", "default", id="default"),
        pytest.param("Default", "default", id="default-upper"),
        pytest.param("off", "default", id="off-alias"),
        pytest.param("OFF", "default", id="off-upper"),
        pytest.param("  off ", "default", id="off-padded"),
    ],
)
def test_the_env_spellings_fold_to_a_request(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    """Case-insensitive and whitespace-trimmed; ``off`` folds onto ``default``.

    ``off`` is the env-friendly alias of ``default`` -- the spelling that means "no
    derived flag" -- so it answers ``default`` rather than being refused.
    """
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, raw)
    assert batch_knobs.resolve_dispatch_formation() == expected


def test_the_explicit_argument_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``explicit argument > environment > default``, in both directions."""
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "aggressive")
    assert batch_knobs.resolve_dispatch_formation() == "aggressive"
    assert batch_knobs.resolve_dispatch_formation("default") == "default"
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "default")
    assert batch_knobs.resolve_dispatch_formation() == "default"
    assert batch_knobs.resolve_dispatch_formation("aggressive") == "aggressive"
    assert batch_knobs.resolve_dispatch_formation("off") == "default"


def test_the_knob_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived process (benchmark, notebook) sees a knob change without a restart."""
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "aggressive")
    assert batch_knobs.resolve_dispatch_formation() == "aggressive"
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "default")
    assert batch_knobs.resolve_dispatch_formation() == "default"
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "auto")
    assert batch_knobs.resolve_dispatch_formation() == "auto"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("junk", id="junk"),
        pytest.param("1", id="int-string"),
        pytest.param("true", id="true"),
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param("none", id="none-of-bands"),
        pytest.param("flat", id="flat-of-bands"),
    ],
)
def test_an_unusable_value_is_a_value_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    """A malformed VALUE is refused loudly, from the env and from the argument.

    ``none`` / ``flat`` belong to ``EVOGP_BATCH_BANDS`` and mean nothing here, and a
    bool is rejected instead of being read as an int -- guessing would silently pin
    a compile flag.  The message names the variable, because the fix is "unset it"
    or "fix the call site".
    """
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, str(value).lower())
    with pytest.raises(ValueError) as excinfo:
        batch_knobs.resolve_dispatch_formation()
    assert batch_knobs.DISPATCH_FORMATION_ENV in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        batch_knobs.resolve_dispatch_formation(value)
    assert batch_knobs.DISPATCH_FORMATION_ENV in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_argument_is_refused_before_the_environment_is_read(
    monkeypatch: pytest.MonkeyPatch, value: bool
) -> None:
    """A bool is its own refusal -- it never borrows the environment's answer.

    The environment is set to a VALID spelling, so a knob that read the env first
    would happily answer ``aggressive``; the bool is refused, and the message names
    the variable (the fix is the call site, not the environment).
    """
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "aggressive")
    with pytest.raises(ValueError) as excinfo:
        batch_knobs.resolve_dispatch_formation(value)
    message = str(excinfo.value)
    assert batch_knobs.DISPATCH_FORMATION_ENV in message, message
    assert repr(value) in message, message


# =========================================================================== #
# 2. the policy: spelling x backend x target x feature detection
# =========================================================================== #
@pytest.mark.parametrize(
    "spelling,backend,gpu,supported,expected",
    [
        # `auto`: aggressive needs ALL THREE (iree, a GPU target, a known option).
        pytest.param("auto", "iree", True, True, "aggressive", id="auto-iree-gpu-known"),
        pytest.param("auto", "iree", True, False, "default", id="auto-old-evoxc"),
        pytest.param("auto", "iree", False, True, "default", id="auto-cpu-target"),
        pytest.param("auto", "iree", False, False, "default", id="auto-cpu-old-evoxc"),
        pytest.param("auto", "numpy", True, True, "default", id="auto-numpy"),
        pytest.param("auto", None, True, True, "default", id="auto-no-backend"),
        pytest.param("auto", "", True, True, "default", id="auto-empty-backend"),
        # an explicit `aggressive` does not need the GPU test (the caller asked),
        # but still needs iree AND an evoxc that knows the option.
        pytest.param(
            "aggressive", "iree", True, True, "aggressive", id="forced-iree-known"
        ),
        pytest.param(
            "aggressive", "iree", False, True, "aggressive", id="forced-cpu-target"
        ),
        pytest.param(
            "aggressive", "iree", True, False, "default", id="forced-old-evoxc"
        ),
        pytest.param("aggressive", "numpy", True, True, "default", id="forced-numpy"),
        pytest.param(
            "aggressive", "numpy", False, False, "default", id="forced-numpy-plain"
        ),
        pytest.param(
            "aggressive", None, True, True, "default", id="forced-no-backend"
        ),
        # `default` / `off` never derive a flag, whatever the target offers.
        pytest.param("default", "iree", True, True, "default", id="default-iree-gpu"),
        pytest.param("off", "iree", True, True, "default", id="off-iree-gpu"),
        pytest.param("default", "numpy", False, False, "default", id="default-numpy"),
    ],
)
def test_the_policy_table(
    spelling: str, backend: Optional[str], gpu: bool, supported: bool, expected: str
) -> None:
    """``effective_dispatch_formation``: the GPU-only enable rule and its degradations.

    The knob's core promise is that an explicit ``aggressive`` (or ``auto``) never
    turns into a failed plan: an evoxc without the option, a non-iree backend and a
    CPU target all answer ``default``.  Only ``iree`` + ``supported`` (+ ``gpu`` for
    ``auto``) can answer ``aggressive``.
    """
    assert (
        batch_knobs.effective_dispatch_formation(
            spelling, backend=backend, gpu=gpu, supported=supported
        )
        == expected
    )
    # The answer is always one of the two effective spellings -- never the request.
    assert expected in (
        batch_knobs.DISPATCH_FORMATION_AGGRESSIVE,
        batch_knobs.DISPATCH_FORMATION_DEFAULT,
    )


def test_a_none_spelling_resolves_the_environment_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spelling=None`` means "resolve the knob here", not "assume a default"."""
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "aggressive")
    assert (
        batch_knobs.effective_dispatch_formation(
            None, backend="iree", gpu=False, supported=True
        )
        == "aggressive"
    )
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "off")
    assert (
        batch_knobs.effective_dispatch_formation(
            None, backend="iree", gpu=True, supported=True
        )
        == "default"
    )
    monkeypatch.delenv(batch_knobs.DISPATCH_FORMATION_ENV)
    assert (
        batch_knobs.effective_dispatch_formation(
            None, backend="iree", gpu=True, supported=True
        )
        == "aggressive"
    )


@pytest.mark.parametrize("spelling", ["AGGRESSIVE", "bogus", "  auto", "None"])
def test_an_unknown_request_is_read_as_the_default_request(spelling: str) -> None:
    """A request ``resolve_*`` can never produce is read as ``auto``, never raised.

    ``effective_dispatch_formation`` is the degradation layer: it only ever answers
    an effective spelling, so a raw spelling that is not one of the three (only
    reachable by bypassing ``resolve_dispatch_formation``) falls back to the
    DEFAULT request -- here ``auto``, which then applies the GPU-only rule.
    """
    aggressive_looking = batch_knobs.effective_dispatch_formation(
        spelling, backend="iree", gpu=True, supported=True
    )
    default_looking = batch_knobs.effective_dispatch_formation(
        spelling, backend="numpy", gpu=False, supported=False
    )
    assert aggressive_looking == "aggressive"
    assert default_looking == "default"


# =========================================================================== #
# 3. the plan's own answer (target policy through the band policy's device kind)
# =========================================================================== #
@pytest.mark.parametrize("kind", ["rocm", "cuda"])
def test_a_gpu_target_with_a_knowing_evoxc_selects_aggressive(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """``plan_dispatch_formation``: ``("auto", "aggressive")`` on a GPU HAL.

    The device kind is read off the SAME ``resolve_band_policy`` the merge uses, so
    the compile flag can never describe a different target than the banded merge.
    The probe answer is forced here (the real one is the installed evoxc's, pinned
    in section 4) so the policy is identical on every checkout.
    """
    monkeypatch.setattr(
        batch_chunks, "supports_dispatch_formation", lambda name="", **_kw: True
    )
    requested, effective = batch_programs.plan_dispatch_formation(
        "iree", SimpleNamespace(kind=kind)
    )
    assert (requested, effective) == ("auto", "aggressive")


def test_a_cpu_target_keeps_the_default_formation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llvm-cpu is flat-to-marginally-worse with the flag, so ``auto`` never sends it.

    The probe is forced to ``True`` here: even a KNOWING evoxc must not get the flag
    on a CPU target, which is exactly what the ``gpu`` half of the rule is for.
    """
    monkeypatch.setattr(
        batch_chunks, "supports_dispatch_formation", lambda name="", **_kw: True
    )
    requested, effective = batch_programs.plan_dispatch_formation(
        "iree", SimpleNamespace(kind="cpu")
    )
    assert (requested, effective) == ("auto", "default")


def test_the_numpy_backend_never_consults_the_feature_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """numpy documents options as accepted-and-ignored, so the key is never put there.

    The probe is replaced by a hard failure: reaching it would mean the route asks
    the iree adapter about a numpy compile (and would import it for nothing).
    """

    def boom(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("the feature probe was consulted for a non-iree plan")

    monkeypatch.setattr(batch_chunks, "supports_dispatch_formation", boom)
    for backend, kind in (("numpy", "rocm"), ("numpy", "cpu"), (None, "cuda")):
        requested, effective = batch_programs.plan_dispatch_formation(
            backend, SimpleNamespace(kind=kind)
        )
        assert (requested, effective) == ("auto", "default"), (backend, kind)


def test_the_request_travels_unchanged_through_the_plan_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first element is the REQUEST, the second what the compile receives.

    A record has to be able to say whether the target policy chose the flag or the
    caller did, so both halves are reported -- for the forced and the off spellings.
    """
    monkeypatch.setattr(
        batch_chunks, "supports_dispatch_formation", lambda name="", **_kw: True
    )
    device = SimpleNamespace(kind="rocm")
    assert batch_programs.plan_dispatch_formation("iree", device, "default") == (
        "default",
        "default",
    )
    assert batch_programs.plan_dispatch_formation("iree", device, "off") == (
        "default",
        "default",
    )
    assert batch_programs.plan_dispatch_formation("iree", device, "aggressive") == (
        "aggressive",
        "aggressive",
    )
    assert batch_programs.plan_dispatch_formation("iree", device, "auto") == (
        "auto",
        "aggressive",
    )
    # ... and with an evoxc that does not know the option, the forced request
    # degrades but the request itself is still recorded verbatim.
    monkeypatch.setattr(
        batch_chunks, "supports_dispatch_formation", lambda name="", **_kw: False
    )
    assert batch_programs.plan_dispatch_formation("iree", device, "aggressive") == (
        "aggressive",
        "default",
    )


def test_an_unusable_value_still_raises_from_the_plan_helper() -> None:
    """The plan helper is not a degradation layer for VALUES: it refuses them."""
    with pytest.raises(ValueError) as excinfo:
        batch_programs.plan_dispatch_formation(
            "iree", SimpleNamespace(kind="rocm"), "junk"
        )
    assert batch_knobs.DISPATCH_FORMATION_ENV in str(excinfo.value)


def test_the_real_probe_decides_the_gpu_rule_end_to_end() -> None:
    """The installed evoxc answers for real: GPU + known option -> ``aggressive``.

    Skipped on an evoxc checkout that predates the option (the route then records
    ``("auto", "default")`` -- asserted by the sibling test below).
    """
    _require_option()
    for kind in ("rocm", "cuda"):
        assert batch_programs.plan_dispatch_formation(
            "iree", SimpleNamespace(kind=kind)
        ) == ("auto", "aggressive")
    assert batch_programs.plan_dispatch_formation(
        "iree", SimpleNamespace(kind="cpu")
    ) == ("auto", "default")
    assert batch_programs.plan_dispatch_formation(
        "numpy", SimpleNamespace(kind="rocm")
    ) == ("auto", "default")


def test_an_evoxc_without_the_option_degrades_instead_of_failing() -> None:
    """The other half of the rule: without the option, even a GPU plan says ``default``.

    This is the plain-checkout behaviour (and the forward-compatible one), so it is
    asserted only where the installed evoxc really lacks the option.
    """
    if batch_chunks.supports_dispatch_formation("iree"):
        pytest.skip("the installed evoxc declares the option; see the test above")
    for value in (None, "aggressive"):
        assert batch_programs.plan_dispatch_formation(
            "iree", SimpleNamespace(kind="rocm"), value
        ) == (value or "auto", "default")


# =========================================================================== #
# 4. the feature probe itself
# =========================================================================== #
def test_the_probe_answers_no_for_a_backend_without_an_adapter() -> None:
    """``numpy`` (and an unknown name) has no optional adapter -> ``False``, no import."""
    assert batch_chunks.supports_dispatch_formation("numpy") is False
    assert batch_chunks.supports_dispatch_formation("numpy", refresh=True) is False
    assert batch_chunks.supports_dispatch_formation("") is False
    # The adapter walk itself is what answers None for a name with no adapter.
    assert batch_chunks._adapter_compile_options("numpy") is None
    assert batch_chunks._adapter_compile_options("not-a-backend") is None


@pytest.mark.parametrize(
    "options,expected",
    [
        pytest.param(None, False, id="no-adapter-table"),
        pytest.param(set(), False, id="empty-table"),
        pytest.param({"opt_level", "target_backends"}, False, id="option-absent"),
        pytest.param({"opt_level", "dispatch_formation"}, True, id="option-present"),
        pytest.param({"dispatch_formation"}, True, id="option-only"),
    ],
)
def test_the_probe_reads_the_adapters_compile_option_table(
    monkeypatch: pytest.MonkeyPatch, options: Any, expected: bool
) -> None:
    """The answer is membership in ``KNOWN_OPTIONS["compile"]``, nothing else.

    That is the table ``evoxc.compile`` validates an options dict against, so the
    probe cannot drift from what a build would accept -- the reason it walks the
    adapter module instead of guessing from a version string.
    """
    monkeypatch.setattr(
        batch_chunks, "_adapter_compile_options", lambda name: options
    )
    assert batch_chunks.supports_dispatch_formation("iree") is expected


def test_the_probe_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throwing walk is "not supported", not an error: it decides an OPTIONAL flag."""
    calls = {"n": 0}

    def boom(name: str) -> Any:
        calls["n"] += 1
        raise RuntimeError("a surprise from the adapter registry")

    monkeypatch.setattr(batch_chunks, "_adapter_compile_options", boom)
    assert batch_chunks.supports_dispatch_formation("iree") is False
    assert batch_chunks.supports_dispatch_formation("iree", refresh=True) is False
    assert calls["n"] == 2, "refresh must really re-probe"


def test_the_probe_is_memoized_and_refresh_re_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One registry walk per backend per process; ``refresh=True`` re-walks and re-caches."""
    state: Dict[str, Any] = {"options": {"dispatch_formation"}, "calls": 0}

    def fake(name: str) -> Any:
        state["calls"] += 1
        return state["options"]

    monkeypatch.setattr(batch_chunks, "_adapter_compile_options", fake)
    assert batch_chunks.supports_dispatch_formation("iree") is True
    assert state["calls"] == 1
    # The memo answers: the option table is static, so the walk is paid once.
    assert batch_chunks.supports_dispatch_formation("iree") is True
    assert state["calls"] == 1
    state["options"] = set()
    assert batch_chunks.supports_dispatch_formation("iree") is True, "not memoized"
    assert state["calls"] == 1
    # ... and a refresh replaces the answer (and caches the new one).
    assert batch_chunks.supports_dispatch_formation("iree", refresh=True) is False
    assert state["calls"] == 2
    assert batch_chunks.supports_dispatch_formation("iree") is False
    assert state["calls"] == 2, "the refreshed answer must be cached"
    # Every backend name is its OWN memo entry (the walk itself is what answers
    # ``None`` for a name without an adapter, so the probe asks it either way).
    assert batch_chunks.supports_dispatch_formation("numpy") is False
    assert state["calls"] == 3
    state["options"] = {"dispatch_formation"}
    assert batch_chunks.supports_dispatch_formation("numpy") is False
    assert state["calls"] == 3, "numpy's answer was not memoized on its own name"
    assert batch_chunks.supports_dispatch_formation("numpy", refresh=True) is True
    assert state["calls"] == 4
    # ... and iree keeps ITS cached answer through numpy's refresh.
    assert batch_chunks.supports_dispatch_formation("iree") is False
    assert state["calls"] == 4


def test_the_probe_does_not_import_iree() -> None:
    """The whole point of walking the adapter MODULE: no iree runtime in the process.

    ``evoxc.backends.get(name)`` would activate the adapter, i.e. import the full
    ``iree`` / ``iree.compiler`` / ``iree._runtime_libs`` stack (seconds) just to
    read a static option table.  The probe must stay free of that.
    """
    if "iree" in sys.modules:
        pytest.skip("iree is already imported in this process (another test's leg)")
    batch_chunks.supports_dispatch_formation("iree", refresh=True)
    assert "iree" not in sys.modules, (
        "the probe activated the iree adapter and imported the whole runtime"
    )
    assert "iree.compiler" not in sys.modules


def test_the_adapter_walk_finds_the_option_on_this_checkout() -> None:
    """The real walk agrees with the real table (or the option is genuinely absent)."""
    options = batch_chunks._adapter_compile_options("iree")
    if options is None:
        pytest.skip("this evoxc has no OPTIONAL_ADAPTERS registry")
    assert isinstance(options, set) and options, options
    assert batch_chunks.supports_dispatch_formation("iree") is (
        "dispatch_formation" in options
    )


# =========================================================================== #
# 5. the compile seam: what build_bodies actually receives
# =========================================================================== #
def test_an_aggressive_plan_passes_the_option_to_the_single_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_batch`` -> ``build_bodies(options=...dispatch_formation="aggressive")``.

    The plan's resolved spelling is authoritative (``_build_options``), so the flag
    reaches the compile without the caller passing anything at build time.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    aggressive = dataclasses.replace(
        plan, dispatch_formation=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
    )
    capture = _OptionsCapture()
    capture.install(monkeypatch)
    E.clear_cache()
    B.build_batch(aggressive, backend=BACKEND)
    options = capture.options()
    assert options["dispatch_formation"] == "aggressive", options
    # one merged graph -> one body, and the plan's report is untouched by the build
    assert len(capture.merges()) == 1
    assert len(aggressive.chunks) == 1


def test_a_default_plan_passes_no_dispatch_formation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``default`` adds NOTHING: the build stays byte-identical to the pre-knob one.

    Compared against the engine's own resolved options (one environment read, no
    key), which is what every memo/artifact key is derived from.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    assert plan.dispatch_formation == batch_knobs.DISPATCH_FORMATION_DEFAULT
    resolved = batch_build._resolve_route_backend(BACKEND, None)[2]
    capture = _OptionsCapture()
    capture.install(monkeypatch)
    E.clear_cache()
    B.build_batch(plan, backend=BACKEND)
    options = capture.options()
    assert "dispatch_formation" not in options, options
    assert options == resolved, (options, resolved)


def test_a_chunked_plan_carries_the_option_into_every_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunked path shares ONE options dict, so the flag reaches every body.

    Chunking is a compilation split, not a second policy: a body compiled with the
    flag and a body without it would be two different codegen settings in one call.
    """
    monkeypatch.setattr(batch_knobs, "MIN_CHUNK_FUNCTIONS", 1)
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), compile_workers=2)
    assert len(plan.chunks) == 2, plan.chunk_sizes
    aggressive = dataclasses.replace(
        plan, dispatch_formation=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
    )
    capture = _OptionsCapture()
    capture.install(monkeypatch)
    E.clear_cache()
    built = B.build_batch(aggressive, backend=BACKEND)
    options = capture.options()
    assert options["dispatch_formation"] == "aggressive", options
    assert len(capture.merges()) == 2, "both bodies must be in the one build call"
    assert len(built.executables) == 2
    # ... and a chunked `default` plan still adds no key at all.
    E.clear_cache()
    capture.calls.clear()
    B.build_batch(plan, backend=BACKEND)
    assert "dispatch_formation" not in capture.options()


def test_build_options_copies_instead_of_mutating_the_callers_dict() -> None:
    """The caller's options (and every key derived from them) stay per-call immutable."""
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    aggressive = dataclasses.replace(
        plan, dispatch_formation=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
    )
    mine = {"opt_level": "O0"}
    out = batch_build._build_options(aggressive, mine)
    assert out == {"opt_level": "O0", "dispatch_formation": "aggressive"}
    assert out is not mine
    assert mine == {"opt_level": "O0"}, "the caller's dict was mutated"
    # A `default` plan answers an equal-but-separate dict (never the same object).
    plain = batch_build._build_options(plan, mine)
    assert plain == mine
    assert plain is not mine
    assert mine == {"opt_level": "O0"}


# =========================================================================== #
# 6. identity: the spelling is in the plan key and in the executable key
# =========================================================================== #
def test_the_spelling_separates_plans_and_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two spellings -> two plan entries and two compiled bodies; a repeat hits.

    The spelling is in BOTH memo keys (the plan key through the resolved effective
    value, the executable key through the frozen compile options), because serving
    one spelling's body to the other request would run codegen nobody asked for.
    """
    values, types, sizes = _population()
    x = _x()
    gate = _Gate()
    gate.install(monkeypatch)
    E.clear_cache()

    gate.effective = batch_knobs.DISPATCH_FORMATION_DEFAULT
    B.evaluate_batch(values, types, sizes, INPUT_LEN, 1, x, per_row=False,
                     backend=BACKEND)
    first = B.batch_cache_stats()
    assert first["batch_plan_misses"] == 1 and first["batch_exec_misses"] == 1

    gate.effective = batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
    B.evaluate_batch(values, types, sizes, INPUT_LEN, 1, x, per_row=False,
                     backend=BACKEND)
    second = B.batch_cache_stats()
    assert second["batch_plan_misses"] == 2, "a re-spelt option must re-plan"
    assert second["batch_plan_entries"] == 2
    assert second["batch_exec_misses"] == 2, "a re-spelt option must re-compile"
    assert second["batch_exec_entries"] == 2
    assert second["batch_plan_hits"] == 0 and second["batch_exec_hits"] == 0

    # A repeat of the SAME spelling is an identity-level plan hit: nothing rebuilds.
    gate.effective = batch_knobs.DISPATCH_FORMATION_DEFAULT
    B.evaluate_batch(values, types, sizes, INPUT_LEN, 1, x, per_row=False,
                     backend=BACKEND)
    third = B.batch_cache_stats()
    assert third["batch_plan_hits"] == 1
    assert third["batch_plan_misses"] == 2
    assert third["batch_exec_misses"] == 2, "the hit must not reach a build"
    assert third["batch_exec_entries"] == 2
    # The spelling is resolved on EVERY call (it is in the plan key), and a plan
    # that is really built resolves it once more to record it: 3 calls + 2 plans.
    assert gate.calls == 5, (
        "the spelling must be resolved per call and per real plan, not cached"
    )


def test_the_executable_key_depends_on_the_frozen_options() -> None:
    """``_executable_key`` differs between the two option dicts (the second layer)."""
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    aggressive = dataclasses.replace(
        plan, dispatch_formation=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE
    )
    plain_key = batch_keys._executable_key(
        plan, BACKEND, plan.device, batch_build._build_options(plan, {})
    )
    aggressive_key = batch_keys._executable_key(
        aggressive, BACKEND, plan.device, batch_build._build_options(aggressive, {})
    )
    assert plain_key != aggressive_key
    assert "dispatch_formation" not in dict(plain_key[-1]), plain_key[-1]
    assert ("dispatch_formation", "aggressive") in tuple(aggressive_key[-1]), (
        aggressive_key[-1]
    )
    # The frozen options are the only difference: the plans' bodies are the same.
    assert plain_key[:-1] == aggressive_key[:-1]


# =========================================================================== #
# 7. the report: both spellings on the plan and in describe_plan()
# =========================================================================== #
def test_the_plan_records_both_spellings() -> None:
    """``BatchPlan.dispatch_formation`` / ``.dispatch_formation_requested``."""
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    assert plan.dispatch_formation == batch_knobs.DISPATCH_FORMATION_DEFAULT
    assert plan.dispatch_formation_requested == batch_knobs.DISPATCH_FORMATION_AUTO
    forced = _plan(values, types, sizes, _x(), dispatch_formation="aggressive")
    # numpy can never receive the flag, so the effective value degrades while the
    # request is recorded verbatim.
    assert forced.dispatch_formation == batch_knobs.DISPATCH_FORMATION_DEFAULT
    assert forced.dispatch_formation_requested == "aggressive"
    off = _plan(values, types, sizes, _x(), dispatch_formation="off")
    assert off.dispatch_formation == batch_knobs.DISPATCH_FORMATION_DEFAULT
    assert off.dispatch_formation_requested == batch_knobs.DISPATCH_FORMATION_DEFAULT


def test_describe_plan_exposes_the_spellings_without_losing_a_key() -> None:
    """The report keeps its whole documented key set and reports the two fields.

    ``describe_plan`` is what every test and benchmark record reads a plan from, so
    a lost key is a silent hole in every record; the set is pinned exactly (not as
    a subset) for that reason.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    info = B.describe_plan(plan)
    assert set(info) == _DESCRIBE_KEYS, sorted(set(info) ^ _DESCRIBE_KEYS)
    assert info["dispatch_formation"] == "default"
    assert info["dispatch_formation_requested"] == "auto"
    assert info["dispatch_formation"] == plan.dispatch_formation
    assert info["dispatch_formation_requested"] == plan.dispatch_formation_requested
    forced = B.describe_plan(
        dataclasses.replace(
            plan,
            dispatch_formation=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE,
            dispatch_formation_requested=batch_knobs.DISPATCH_FORMATION_AGGRESSIVE,
        )
    )
    assert set(forced) == _DESCRIBE_KEYS
    assert forced["dispatch_formation"] == "aggressive"
    assert forced["dispatch_formation_requested"] == "aggressive"
    # every other field is the same plan's: only the option seam moved
    differing = {key for key in info if info[key] != forced[key]}
    assert differing == {
        "dispatch_formation",
        "dispatch_formation_requested",
    }, differing


def test_the_environment_reaches_the_plan_through_the_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVOGP_BATCH_DISPATCH_FORMATION`` alone moves the recorded request.

    An operator sets the variable and restarts nothing: the next plan records it,
    which is what makes the knob usable from a benchmark script.
    """
    values, types, sizes = _population()
    baseline = B.describe_plan(_plan(values, types, sizes, _x()))
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "off")
    off = B.describe_plan(_plan(values, types, sizes, _x()))
    assert off["dispatch_formation_requested"] == "default"
    assert off["dispatch_formation"] == baseline["dispatch_formation"] == "default"
    monkeypatch.setenv(batch_knobs.DISPATCH_FORMATION_ENV, "AGGRESSIVE")
    forced = B.describe_plan(_plan(values, types, sizes, _x()))
    assert forced["dispatch_formation_requested"] == "aggressive"
    # numpy can never receive the flag: the effective spelling stays `default`.
    assert forced["dispatch_formation"] == "default"
