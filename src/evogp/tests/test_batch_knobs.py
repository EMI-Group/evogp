"""Knob, band-selection and cache/stats surface of the opt-in "batch" route.

``src/evogp/tree/batch_engine.py`` is the THIRD evaluation route ("one merged
``evoxc`` graph per population", reached by ``EVOGP_EVAL_MODE=batch``): it traces
every distinct program of the population, merges those graphs into one executable
with ``evoxc.batch_functions`` and evaluates the whole population with one
``evoxc.run``.
Everything a caller can steer is a ``EVOGP_BATCH_*`` environment variable that is
read PER CALL, with an explicit argument always winning -- and every such knob is
easy to break silently, because a knob that stops being read (or is read once at
import time by accident) changes a graph's shape without changing a single test
number.
This file pins the SURFACE of those knobs, not the numbers:

* the five resolvers (``resolve_band_size`` / ``resolve_bands`` /
  ``resolve_switch_ops`` / ``resolve_alignment`` / ``resolve_min_pop``) -- their
  defaults, the explicit-argument-wins rule, the per-call environment read, the
  case/whitespace folding where the module documents it, and one ``ValueError``
  naming the offending variable for every invalid spelling;
* the band-count PRECEDENCE (``switch_ops`` -> explicit ``bands`` -> the
  ``ceil(n / band_size)`` model, clamped into ``1 <= K <= n``) including the
  out-of-range and non-positive refusals;
* the resolved merge SHAPE reaching a real plan (``plan_batch`` is pure host work
  -- decode, dedup, trace, merge -- so introspection costs no compile), through
  both the explicit arguments and the environment;
* the two-level front table (``BATCH_PLAN_MEMO`` / ``BATCH_EXEC_MEMO``), its exact
  8-key ``batch_cache_stats()`` contract, and the identity/content semantics that
  make a repeat call an identity-level plan hit, a REBUILT-equal population a
  content-level hit (no re-plan, no re-compile), and an unchanged program multiset
  at a DIFFERENT population size an executable hit;
* the disjointness of those counters from ``cache_stats()`` (exactly its own 8
  keys), ``dense_cache_stats()``, ``artifact_cache_stats()`` and the content
  table's own 5-key ``batch_content_stats()`` -- four surfaces other files in this
  suite pin, and none of which may grow a ``batch_*`` key;
* the decline contract: ``BatchUnsupported`` (a ``RuntimeError``) for "not my
  population" (below ``EVOGP_BATCH_MIN_POP``, empty, an unknown backend name, an
  ``x``/``per_row`` shape doubt) and ``ValueError`` for "your configuration is
  malformed" (a bad knob, an unknown variant).

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_engine.py`` owns the NUMERIC contract (merged graph vs the
  specialized engine and vs the independent plain-torch oracle, both variants,
  multi-output, NaN/+-inf placement, row expansion) and the band spellings'
  band-count/strategy table at the value level.
* ``test_batch_route.py`` owns the PUBLIC-ROUTE wiring (which of the five call
  sites takes the batch branch, and the ``BatchUnsupported`` fallback), and
  ``test_dense_policy.py`` / ``test_route_policy.py`` own the route POLICY
  (``EVOGP_EVAL_MODE=batch``, the ``Route`` fields and the engine tables). This
  module never consults either: neither ``plan_batch`` / ``evaluate_batch`` /
  ``sr_fitness_batch`` nor any resolver here reads ``EVOGP_EVAL_MODE``, so the
  whole file behaves identically under the default run and under a forced
  ``EVOGP_EVAL_MODE=dense`` / ``=batch`` run.
* ``test_dense_host_memo.py`` owns the dense front tables; this file only proves
  the batch tables are invisible to them.

Mechanics
---------
* Hermetic numpy ONLY: every engine call passes ``backend="numpy"`` explicitly
  (``conftest.py`` pins ``EVOGP_EVOXC_BACKEND=numpy`` on top), no iree package is
  needed and no device is touched.
* No wall-clock assertion anywhere -- the checks are values, shapes, flags,
  plan reports and cache counters.
* Tiny shapes (pop 4 out of 4 distinct programs, ``N = 5`` rows, ``L = 16``), so
  the handful of numpy compiles stay milliseconds and the file runs in a few
  seconds; the tiny populations are admitted through ``EVOGP_BATCH_MIN_POP=1``
  (the library default of 64 is the measured break-even, i.e. a real decline).
* The autouse fixture clears every knob (the merge knobs, the population
  threshold and the compile-worker count) and brackets every test with
  ``evoxc_engine.clear_cache()`` (the single in-memory invalidation point, which
  resets every batch table -- plan, executable and content -- through the
  registered hook).
* ``EVOGP_BATCH_CACHE_SIZE`` is read at IMPORT time -- the tables are built once,
  so this file only PINS the value they report and never tries to change it.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_knobs.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Iterator, Sequence, Tuple

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

from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_knobs  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: this file is about knobs, not economics)
# --------------------------------------------------------------------------- #
#: The host backend: nothing compiles to a device, nothing is uploaded.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: FOUR distinct programs -- three computed ones and one fully constant tree, i.e.
#: the population shape a GP loop presents and the smallest one with a non-trivial
#: band count.  Distinct, so the merged function count equals the population size.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
POP_SIZE = len(_PROGRAMS)
#: The four merge knobs plus the threshold and the compile-worker count: all of
#: them are cleared per test, so an ambient value can never change the plan a test
#: asks for.  The compile-worker knob is read through ``getattr`` so this module
#: still COLLECTS while the tree layer that defines it is landing in parallel.
_KNOB_ENVS: Tuple[str, ...] = (
    B.BAND_SIZE_ENV,
    B.BANDS_ENV,
    B.SWITCH_OPS_ENV,
    B.ALIGNMENT_ENV,
    B.MIN_POP_ENV,
    getattr(batch_knobs, "COMPILE_WORKERS_ENV", "EVOGP_BATCH_COMPILE_WORKERS"),
)
#: The exact key set of ``batch_cache_stats()``: two ``dense_memo.DenseMemo``
#: tables, four counters each.  Pinned as a set so an extra/missing key fails.
_BATCH_STAT_KEYS = frozenset(
    {
        "batch_plan_hits",
        "batch_plan_misses",
        "batch_plan_entries",
        "batch_plan_maxsize",
        "batch_exec_hits",
        "batch_exec_misses",
        "batch_exec_entries",
        "batch_exec_maxsize",
    }
)
#: The exact key set of ``batch_content_stats()`` -- the CONTENT level of the
#: two-level plan key (a separate table, a separate function, a separate surface).
_CONTENT_STAT_KEYS = frozenset(
    {
        "batch_content_hits",
        "batch_content_misses",
        "batch_content_entries",
        "batch_content_maxsize",
        "batch_content_rejects",
    }
)
#: ``cache_stats()``'s documented 8 keys (the engine's own contract, pinned here
#: only to prove the batch tables did not leak into it).
_ENGINE_STAT_KEYS = frozenset(
    {
        "hits",
        "misses",
        "entries",
        "maxsize",
        "memo_hits",
        "memo_misses",
        "memo_entries",
        "memo_maxsize",
    }
)


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear every knob and bracket every test with empty batch tables.

    The knobs are DELETED (not pinned to a default) so "the default is 32 / 64 /
    ``variant`` / off" means the same thing on a box that exports them; each test
    then sets what it needs on the same ``monkeypatch`` and pytest restores the
    ambient values at teardown.
    ``evoxc_engine.clear_cache()`` is the single in-memory invalidation point and
    resets every batch table (plan, executable, content) through their registered
    hook, so every counter assertion below starts from zero and no entry outlives
    its test.
    """
    for name in _KNOB_ENVS:
        monkeypatch.delenv(name, raising=False)
    E.clear_cache()
    try:
        yield
    finally:
        E.clear_cache()


@pytest.fixture
def tiny_pop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit this file's tiny populations.

    ``EVOGP_BATCH_MIN_POP`` defaults to 64 (the measured call-count break-even), and
    the three entry points have no ``min_pop=`` argument, so the environment is the
    only way to admit a 4-row population -- exactly what a caller raising the knob
    on purpose does.
    """
    monkeypatch.setenv(B.MIN_POP_ENV, "1")


def _population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``(values, types, sizes)`` CPU population tensors of ``_PROGRAMS``.

    A fresh call allocates FRESH buffers, which is what the plan-memo identity
    semantics are about: same content, different identity tokens.
    """
    values, types, sizes = te.rows_from_specs(
        _PROGRAMS, output_len=1, pad_to=ROW_LEN, tail_value=0.0
    )
    return values, types, sizes


def _constant_rows(pop_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An all-constant population: one node, ``3.5``, every row identical.

    Every row therefore collapses onto ONE merged function, which is how the
    cross-population executable-hit test below gets the same program multiset at
    three different population sizes.
    """
    values = torch.zeros((pop_size, 4), dtype=torch.float32)
    values[:, 0] = 3.5
    types = torch.zeros((pop_size, 4), dtype=torch.int16)
    types[:, 0] = te.CONST
    sizes = torch.zeros((pop_size, 4), dtype=torch.int16)
    sizes[:, 0] = 1
    return values, types, sizes


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a local generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int = N_ROWS, seed: int = 101) -> torch.Tensor:
    """Deterministic ``(rows, 1)`` SR labels from a local generator."""
    return te.x_random(rows, 1, seed=seed)


def _plan(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    **knobs: Any,
) -> Any:
    """``plan_batch`` with the shape grid, the numpy backend and the loss variant.

    The loss variant (``per_row=False``) is the layout every band fixture here
    uses: it is the one the training loop spends its compiles on, and its merged
    output is one scalar per function.  ``plan_batch`` never compiles.
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


def _info(plan: Any) -> Dict[str, Any]:
    """The whole ``describe_plan`` report (every CONFIGURATION field of it).

    ``stage_ms`` is dropped: the plan-side stage timings are MEASUREMENT (they
    change on every call and on every machine), not plan content, so two plans
    built from the same knobs can never compare equal on them.  Every other field
    -- including the chunk geometry -- is compared unchanged.
    """
    info = dict(B.describe_plan(plan))
    timing = info.pop("stage_ms", None)
    assert timing is None or isinstance(timing, dict), (
        f"the plan-side stage timings must be a dict when they are reported: {timing!r}"
    )
    return info


def _loss(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    **knobs: Any,
) -> torch.Tensor:
    """``sr_fitness_batch`` with the shape grid and the numpy backend pinned."""
    return B.sr_fitness_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        labels,
        True,
        backend=BACKEND,
        **knobs,
    )


def _value(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    *,
    per_row: bool,
    **knobs: Any,
) -> torch.Tensor:
    """``evaluate_batch`` with the shape grid and the numpy backend pinned."""
    return B.evaluate_batch(
        values, types, sizes, INPUT_LEN, 1, x, per_row=per_row, backend=BACKEND, **knobs
    )


class _Counter:
    """Wrap a module-level callable and count its invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> None:
        original = getattr(module, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", name)
        monkeypatch.setattr(module, name, wrapper)


# =========================================================================== #
# 1. constants, the stats key set, and the import-time cache size
# =========================================================================== #
def test_the_route_constants_are_the_documented_ones() -> None:
    """The module's public names are the seam a wiring task imports.

    Every env name is spelled once here so the autouse clearing fixture (built from
    the same constants) can never target a variable the engine stopped reading.
    """
    assert B.EVAL_MODE == "batch"
    assert B.BAND_SIZE_ENV == "EVOGP_BATCH_BAND_SIZE"
    assert B.BANDS_ENV == "EVOGP_BATCH_BANDS"
    assert B.SWITCH_OPS_ENV == "EVOGP_BATCH_SWITCH_OPS"
    assert B.ALIGNMENT_ENV == "EVOGP_BATCH_ALIGNMENT"
    assert B.MIN_POP_ENV == "EVOGP_BATCH_MIN_POP"
    assert B.CACHE_SIZE_ENV == "EVOGP_BATCH_CACHE_SIZE"
    assert B.DEFAULT_BAND_SIZE == 32
    assert B.DEFAULT_ALIGNMENT == "variant"
    assert B.ALIGNMENTS == ("variant", "occurrence", "auto")
    assert B.DEFAULT_MIN_POP == 64
    assert B.DEFAULT_CACHE_SIZE == 8
    assert (B.VALUE_VARIANT, B.LOSS_VARIANT) == ("value", "loss")
    assert B.PLAN_VARIANTS == (B.VALUE_VARIANT, B.LOSS_VARIANT)


def test_batch_cache_stats_has_exactly_the_documented_keys() -> None:
    """``batch_cache_stats()`` is exactly eight int counters, two tables.

    The key set is a contract: a renamed or added key silently breaks the callers
    that read a delta, and a ``batch_*`` key leaking into ``cache_stats()`` /
    ``dense_cache_stats()`` would break the dense files pinned elsewhere.
    """
    stats = B.batch_cache_stats()
    assert set(stats) == _BATCH_STAT_KEYS, sorted(set(stats) ^ _BATCH_STAT_KEYS)
    assert all(isinstance(value, int) for value in stats.values()), stats
    assert set(B.BATCH_PLAN_MEMO.stats()) == {
        "batch_plan_hits",
        "batch_plan_misses",
        "batch_plan_entries",
        "batch_plan_maxsize",
    }
    assert set(B.BATCH_EXEC_MEMO.stats()) == {
        "batch_exec_hits",
        "batch_exec_misses",
        "batch_exec_entries",
        "batch_exec_maxsize",
    }


def test_the_batch_cache_size_is_read_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVOGP_BATCH_CACHE_SIZE`` bounds the two tables ONCE, when they are built.

    The helper that derives the bound reads the variable per call, but the tables
    themselves were constructed at import time, so setting the variable at runtime
    must not move ``*_maxsize`` -- an operator changing the knob has to restart the
    process (the same rule as ``EVOGP_EVOXC_CACHE_SIZE``), and this pins it instead
    of pretending the knob is live.
    """
    stats = B.batch_cache_stats()
    import_time_maxsize = B.BATCH_PLAN_MEMO.maxsize
    assert stats["batch_plan_maxsize"] == import_time_maxsize
    assert stats["batch_exec_maxsize"] == B.BATCH_EXEC_MEMO.maxsize == import_time_maxsize
    assert import_time_maxsize >= 0
    monkeypatch.setenv(B.CACHE_SIZE_ENV, "999")
    assert B._default_cache_size() == 999, "the helper itself reads the env per call"
    assert B.BATCH_PLAN_MEMO.maxsize == import_time_maxsize, (
        "the plan table was rebuilt from the new env value at runtime"
    )
    assert B.batch_cache_stats()["batch_plan_maxsize"] == import_time_maxsize


@pytest.mark.parametrize("raw", ["-1", "abc", "2.5"])
def test_the_cache_size_helper_rejects_a_malformed_value(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A negative or non-integer bound is a ``ValueError`` naming the variable.

    The bound is only read at import time, so a malformed value normally fails the
    import; the helper keeps the same error contract for a caller re-deriving it.
    """
    monkeypatch.setenv(B.CACHE_SIZE_ENV, raw)
    with pytest.raises(ValueError) as excinfo:
        B._default_cache_size()
    assert B.CACHE_SIZE_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 2. resolve_band_size (the ceil(n / size) model's size knob)
# =========================================================================== #
def test_band_size_default_env_and_explicit_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVOGP_BATCH_BAND_SIZE``: default 32, env next, explicit argument last.

    The explicit argument winning is what lets a caller (or a runner) pin the merge
    shape for one call without touching the process environment.
    """
    assert B.resolve_band_size() == B.DEFAULT_BAND_SIZE == 32
    monkeypatch.setenv(B.BAND_SIZE_ENV, "7")
    assert B.resolve_band_size() == 7
    assert B.resolve_band_size(3) == 3, "the explicit argument must win over the env"
    assert B.resolve_band_size(1) == 1


def test_band_size_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knob is resolved on every call, not memoized at import time.

    A long-lived process (a benchmark script, a notebook) must see a knob change
    without a restart -- and whitespace around the value is trimmed.
    """
    monkeypatch.setenv(B.BAND_SIZE_ENV, "7")
    assert B.resolve_band_size() == 7
    monkeypatch.setenv(B.BAND_SIZE_ENV, "9")
    assert B.resolve_band_size() == 9
    monkeypatch.setenv(B.BAND_SIZE_ENV, " 12 ")
    assert B.resolve_band_size() == 12


@pytest.mark.parametrize(
    "source,value",
    [
        pytest.param("env", "-1", id="env-negative"),
        pytest.param("env", "0", id="env-zero"),
        pytest.param("env", "abc", id="env-not-an-int"),
        pytest.param("arg", 0, id="arg-zero"),
        pytest.param("arg", -3, id="arg-negative"),
    ],
)
def test_band_size_rejects_values_below_one(
    monkeypatch: pytest.MonkeyPatch, source: str, value: Any
) -> None:
    """A band size below 1 cannot be honoured, so it is a ``ValueError``.

    The message names ``EVOGP_BATCH_BAND_SIZE`` from BOTH spellings (the env and the
    argument), because the fix differs: unset the variable, or fix the call site.
    """
    if source == "env":
        monkeypatch.setenv(B.BAND_SIZE_ENV, value)
        call = B.resolve_band_size
    else:
        call = lambda: B.resolve_band_size(value)  # noqa: E731 - one-line helper
    with pytest.raises(ValueError) as excinfo:
        call()
    assert B.BAND_SIZE_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 3. resolve_bands (explicit band count, or the flat spellings)
# =========================================================================== #
def test_bands_defaults_to_the_size_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` means "the band-size model decides" -- for unset/empty only.

    Every flat spelling (``none`` / ``flat`` / ``0``, and the integer ``0``) asks
    for an explicit SINGLE band and answers ``1``: handing it back as ``None``
    would drop the request into the band-size model, i.e. band a merge the caller
    explicitly asked to keep flat.
    """
    assert B.resolve_bands() is None
    monkeypatch.setenv(B.BANDS_ENV, "")
    assert B.resolve_bands() is None, "an empty value must mean 'unset'"
    for spelling in ("none", "flat", "0"):
        assert B.resolve_bands(spelling) == 1, spelling
    assert B.resolve_bands(0) == 1, "an integer 0 is the same 'flat' request"
    assert B.resolve_bands("") is None


def test_bands_returns_an_explicit_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """An integer ``K`` (from the argument or the env) comes back as given.

    A flat spelling is an explicit ``1``, so it comes back as ``1`` too.  Range
    validation against the function count is ``resolve_band_count``'s job: the
    count is resolved before the population is decoded.
    """
    assert B.resolve_bands(3) == 3
    assert B.resolve_bands("3") == 3
    monkeypatch.setenv(B.BANDS_ENV, " 5 ")
    assert B.resolve_bands() == 5
    assert B.resolve_bands(2) == 2, "the explicit argument must win over the env"
    assert B.resolve_bands("none") == 1, "even a flat argument must win"


def test_bands_folds_case_and_whitespace() -> None:
    """The flat spellings are case-insensitive and whitespace-trimmed.

    ``EVOGP_BATCH_BANDS= NONE `` in a shell profile must not become a typo error,
    and must still mean the single flat band (``1``), not "unset" (``None``).
    """
    for spelling in (" NONE ", "Flat", "\tnone\n", " 0 "):
        assert B.resolve_bands(spelling) == 1, repr(spelling)


@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_bands_rejects_a_bool(value: bool) -> None:
    """``bool`` is an ``int``: instead of guessing (``True`` -> 1?) it is refused.

    Silently reading a boolean as a band count is the kind of guess that produces a
    plausible-looking but wrong merge shape, so the resolver names the variable.
    """
    with pytest.raises(ValueError) as excinfo:
        B.resolve_bands(value)
    assert B.BANDS_ENV in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "source,value",
    [
        pytest.param("arg", "maybe", id="arg-text"),
        pytest.param("arg", "1.5", id="arg-float-text"),
        pytest.param("env", "maybe", id="env-text"),
        pytest.param("env", "two", id="env-word"),
    ],
)
def test_bands_rejects_other_text(
    monkeypatch: pytest.MonkeyPatch, source: str, value: str
) -> None:
    """Any other spelling is a ``ValueError`` naming ``EVOGP_BATCH_BANDS``.

    Both spellings are covered: the env read and the explicit argument share one
    message, so a typo is reported the same way wherever it came from.
    """
    if source == "env":
        monkeypatch.setenv(B.BANDS_ENV, value)
        call = B.resolve_bands
    else:
        call = lambda: B.resolve_bands(value)  # noqa: E731 - one-line helper
    with pytest.raises(ValueError) as excinfo:
        call()
    assert B.BANDS_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 4. resolve_switch_ops (the flat op-switching merge)
# =========================================================================== #
def test_switch_ops_defaults_off_and_reads_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVOGP_BATCH_SWITCH_OPS`` is off by default and folded case-insensitively.

    Op-switching is a cost knob that measured as a no-op on the GP workload, so the
    default must stay off; every documented spelling (and only those) is accepted.
    """
    assert B.resolve_switch_ops() is False
    monkeypatch.setenv(B.SWITCH_OPS_ENV, "")
    assert B.resolve_switch_ops() is False, "an empty value must mean 'off'"
    for spelling in ("1", "true", "yes", "on", " TRUE ", "On"):
        monkeypatch.setenv(B.SWITCH_OPS_ENV, spelling)
        assert B.resolve_switch_ops() is True, spelling
    for spelling in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv(B.SWITCH_OPS_ENV, spelling)
        assert B.resolve_switch_ops() is False, spelling


def test_switch_ops_explicit_argument_wins_over_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit argument beats the env in both directions.

    It also bypasses an INVALID env spelling: a call that states its own value must
    not fail on an ambient typo it never reads.
    """
    monkeypatch.setenv(B.SWITCH_OPS_ENV, "1")
    assert B.resolve_switch_ops(False) is False
    monkeypatch.setenv(B.SWITCH_OPS_ENV, "0")
    assert B.resolve_switch_ops(True) is True
    monkeypatch.setenv(B.SWITCH_OPS_ENV, "maybe")
    assert B.resolve_switch_ops(True) is True


@pytest.mark.parametrize("spelling", ["maybe", "2", "yes please", "-1"])
def test_switch_ops_rejects_other_text(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Only the eight documented spellings (and the empty value) are accepted.

    Anything else is a ``ValueError`` naming ``EVOGP_BATCH_SWITCH_OPS``: guessing a
    truthiness would silently change the merge shape.
    """
    monkeypatch.setenv(B.SWITCH_OPS_ENV, spelling)
    with pytest.raises(ValueError) as excinfo:
        B.resolve_switch_ops()
    assert B.SWITCH_OPS_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 5. resolve_alignment (the switch_ops class-alignment heuristic)
# =========================================================================== #
def test_alignment_defaults_and_folds_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EVOGP_BATCH_ALIGNMENT`` defaults to ``variant`` and folds case/space.

    Alignment is a pure performance heuristic inside evoxc -- it can never change
    the numbers -- so a looser spelling is safe; the value is still canonicalized.
    """
    assert B.resolve_alignment() == B.DEFAULT_ALIGNMENT == "variant"
    monkeypatch.setenv(B.ALIGNMENT_ENV, "")
    assert B.resolve_alignment() == B.DEFAULT_ALIGNMENT
    for mode in B.ALIGNMENTS:
        monkeypatch.setenv(B.ALIGNMENT_ENV, f" {mode.upper()} ")
        assert B.resolve_alignment() == mode, mode


def test_alignment_explicit_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``alignment=`` is used as-is, canonicalized, over the env."""
    monkeypatch.setenv(B.ALIGNMENT_ENV, "occurrence")
    assert B.resolve_alignment("auto") == "auto"
    assert B.resolve_alignment(" OCCURRENCE ") == "occurrence"
    assert B.resolve_alignment() == "occurrence", "the env is only the fallback"


@pytest.mark.parametrize(
    "source,value",
    [
        pytest.param("env", "nope", id="env-unknown"),
        pytest.param("env", "variants", id="env-typo"),
        pytest.param("arg", "nope", id="arg-unknown"),
    ],
)
def test_alignment_rejects_an_unknown_spelling(
    monkeypatch: pytest.MonkeyPatch, source: str, value: str
) -> None:
    """An unknown alignment is a ``ValueError`` naming the variable and the value.

    Both spellings are validated identically, so a typo is never turned into a
    silent fallback onto the default.
    """
    if source == "env":
        monkeypatch.setenv(B.ALIGNMENT_ENV, value)
        call = B.resolve_alignment
    else:
        call = lambda: B.resolve_alignment(value)  # noqa: E731 - one-line helper
    with pytest.raises(ValueError) as excinfo:
        call()
    message = str(excinfo.value)
    assert B.ALIGNMENT_ENV in message and value in message, message


def test_alignment_is_validated_even_with_op_switching_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The knob is validated on the DEFAULT path, where it has no consumer.

    Only a ``switch_ops`` merge reads the alignment, but a typo must be caught even
    when op-switching is off -- otherwise it surfaces later, on the one run that
    turns switching on.
    """
    monkeypatch.setenv(B.SWITCH_OPS_ENV, "0")
    monkeypatch.setenv(B.ALIGNMENT_ENV, "align")
    assert B.resolve_switch_ops() is False
    with pytest.raises(ValueError) as excinfo:
        B.resolve_alignment()
    assert B.ALIGNMENT_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 6. resolve_min_pop (the population threshold of the route)
# =========================================================================== #
def test_min_pop_default_env_and_explicit_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVOGP_BATCH_MIN_POP`` defaults to the measured break-even 64.

    Lowering it (env or ``value=``) is how a caller admits a small population on
    purpose -- the entry points have no ``min_pop=`` argument, so the env is the
    supported route and the argument form is the resolver's test seam.
    """
    assert B.resolve_min_pop() == B.DEFAULT_MIN_POP == 64
    monkeypatch.setenv(B.MIN_POP_ENV, "8")
    assert B.resolve_min_pop() == 8
    assert B.resolve_min_pop(4) == 4, "the explicit argument must win over the env"
    assert B.resolve_min_pop(1) == 1
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    assert B.resolve_min_pop() == 1, "the admitted case the fixtures rely on"


@pytest.mark.parametrize(
    "source,value",
    [
        pytest.param("env", "0", id="env-zero"),
        pytest.param("env", "-4", id="env-negative"),
        pytest.param("env", "many", id="env-not-an-int"),
        pytest.param("arg", 0, id="arg-zero"),
        pytest.param("arg", -1, id="arg-negative"),
    ],
)
def test_min_pop_rejects_values_below_one(
    monkeypatch: pytest.MonkeyPatch, source: str, value: Any
) -> None:
    """A threshold below 1 is a ``ValueError`` naming ``EVOGP_BATCH_MIN_POP``.

    ``0`` would decline every population (including a non-empty one); the resolver
    refuses instead of leaving that ambiguity in the route.
    """
    if source == "env":
        monkeypatch.setenv(B.MIN_POP_ENV, value)
        call = B.resolve_min_pop
    else:
        call = lambda: B.resolve_min_pop(value)  # noqa: E731 - one-line helper
    with pytest.raises(ValueError) as excinfo:
        call()
    assert B.MIN_POP_ENV in str(excinfo.value), str(excinfo.value)


# =========================================================================== #
# 7. resolve_band_count (the precedence: switch_ops -> bands -> band size)
# =========================================================================== #
@pytest.mark.parametrize(
    "n_functions,band_size,expected",
    [
        pytest.param(1, 32, 1, id="single-function"),
        pytest.param(10, 32, 1, id="below-one-band-size"),
        pytest.param(32, 32, 1, id="exactly-one-band-size"),
        pytest.param(33, 32, 2, id="one-over"),
        pytest.param(64, 32, 2, id="exactly-two"),
        pytest.param(100, 32, 4, id="ceil-not-floor"),
        pytest.param(4, 1, 4, id="size-one-is-per-function"),
        pytest.param(5, 2, 3, id="odd-count-rounds-up"),
        pytest.param(1000, 1, 1000, id="size-one-at-scale"),
        pytest.param(7, 10 ** 6, 1, id="huge-size-is-flat"),
    ],
)
def test_band_count_follows_the_band_size_model(
    n_functions: int, band_size: int, expected: int
) -> None:
    """``K = ceil(n / band_size)``, clamped into ``1 <= K <= n``.

    ``ceil`` (never floor) and the clamp at both ends are the whole model: a huge
    size must still produce ONE band, a size of 1 exactly one band per function.
    """
    assert B.resolve_band_count(n_functions, band_size=band_size) == expected
    assert 1 <= expected <= n_functions


def test_band_count_is_clamped_into_one_to_n() -> None:
    """Both clamp ends are reachable from the public seam.

    ``band_size=1`` gives ``K = n`` (one band per function) and a size far above the
    function count gives ``K = 1`` (the flat merge) -- neither can fall outside
    ``[1, n]``, which is what the emitter requires.
    """
    assert B.resolve_band_count(4, band_size=1) == 4
    assert B.resolve_band_count(4, band_size=10 ** 6) == 1
    assert B.resolve_band_count(1, band_size=1) == 1
    assert B.resolve_band_count(1, band_size=10 ** 6) == 1


def test_an_explicit_band_count_wins_over_the_band_size_model() -> None:
    """An explicit count beats the size model, even one that contradicts it.

    ``bands=3`` with ``band_size=1`` (which alone would ask for 10 bands) must
    return 3: precedence is ``bands`` before the model, not a blend.
    """
    assert B.resolve_band_count(10, bands=3, band_size=1) == 3
    assert B.resolve_band_count(10, bands="2", band_size=1) == 2
    assert B.resolve_band_count(10, bands=10, band_size=10 ** 6) == 10


def test_band_count_reads_the_knobs_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env spellings reach the model identically to the arguments.

    ``EVOGP_BATCH_BANDS`` wins over ``EVOGP_BATCH_BAND_SIZE``, and removing it hands
    the decision back to the size model -- the documented fallback chain.
    """
    monkeypatch.setenv(B.BANDS_ENV, "2")
    assert B.resolve_band_count(10) == 2
    monkeypatch.setenv(B.BAND_SIZE_ENV, "1")
    assert B.resolve_band_count(10) == 2, "an env band count beats the size model"
    monkeypatch.delenv(B.BANDS_ENV)
    assert B.resolve_band_count(10) == 10, "without bands the size model decides"
    monkeypatch.setenv(B.BAND_SIZE_ENV, "3")
    assert B.resolve_band_count(10) == 4


@pytest.mark.parametrize(
    "n_functions,bands",
    [
        pytest.param(4, 5, id="one-too-many"),
        pytest.param(4, 9, id="far-too-many"),
        pytest.param(4, -1, id="negative"),
        pytest.param(4, "5", id="text-count-too-many"),
    ],
)
def test_band_count_rejects_an_out_of_range_explicit_count(
    n_functions: int, bands: Any
) -> None:
    """An explicit ``K`` outside ``1..n`` is a ``ValueError`` naming the variable.

    It cannot be honoured, so silently clamping would hide a typo (a count that
    looks plausible but describes a different population).
    """
    with pytest.raises(ValueError) as excinfo:
        B.resolve_band_count(n_functions, bands=bands)
    assert B.BANDS_ENV in str(excinfo.value), str(excinfo.value)


def test_switch_ops_forces_a_single_band() -> None:
    """``switch_ops=True`` answers 1 without consulting any band spelling.

    Banding is never combined with op-switching, so the knob short-circuits the
    whole precedence chain -- including an out-of-range explicit count, which is
    then simply irrelevant instead of an error.
    """
    assert B.resolve_band_count(10, bands=5, switch_ops=True) == 1
    assert B.resolve_band_count(10, band_size=1, switch_ops=True) == 1
    assert B.resolve_band_count(10, bands="9", switch_ops=True) == 1
    assert B.resolve_band_count(10, switch_ops=False, bands=5) == 5


@pytest.mark.parametrize("n_functions", [0, -1, -100])
def test_band_count_rejects_a_nonpositive_function_count(n_functions: int) -> None:
    """A merge of zero functions is meaningless: ``ValueError`` before any knob.

    The check runs FIRST, so not even ``switch_ops=True`` (which would answer 1
    without looking) can turn an empty merge into a valid plan.
    """
    with pytest.raises(ValueError) as excinfo:
        B.resolve_band_count(n_functions)
    assert str(n_functions) in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError):
        B.resolve_band_count(n_functions, switch_ops=True)


# =========================================================================== #
# 8. the knobs reaching a real plan (plan_batch: host work only, no compile)
# =========================================================================== #
#: ``(knobs, band count, band strategy)`` for the 4-distinct-program fixture.
#: ``bands=1`` and the band-size model's ``K == 1`` both take the flat emitter, so
#: their strategy reads ``"flat"`` -- only an explicit ``K >= 2`` reads
#: ``"explicit"``.
#: NOTE: this fixture's four functions make the size model answer ``K == 1`` as
#: well, so these cases cannot tell a flat spelling that reaches the emitter from
#: one that fell through to the model; ``WIDE_POP_SIZE`` below is the
#: discriminating shape.
_BAND_PLAN_CASES = (
    pytest.param({}, 1, "flat", id="default-size-model-flat"),
    pytest.param({"bands": 1}, 1, "flat", id="bands-1-collapses-to-flat"),
    pytest.param({"bands": 2}, 2, "explicit", id="bands-2"),
    pytest.param({"bands": 4}, 4, "explicit", id="bands-4"),
    pytest.param({"bands": "none"}, 1, "flat", id="bands-none"),
    pytest.param({"bands": "flat"}, 1, "flat", id="bands-flat"),
    pytest.param({"bands": "0"}, 1, "flat", id="bands-zero"),
    pytest.param({"band_size": 1}, 4, "explicit", id="band-size-1"),
    pytest.param({"band_size": 3}, 2, "explicit", id="band-size-3"),
    pytest.param({"band_size": 32}, 1, "flat", id="band-size-default"),
    pytest.param({"switch_ops": True}, 1, "flat", id="switch-ops"),
)


@pytest.mark.parametrize("knobs,expected_bands,expected_strategy", _BAND_PLAN_CASES)
def test_the_band_knobs_reach_the_plan(
    tiny_pop: None, knobs: Dict[str, Any], expected_bands: int, expected_strategy: str
) -> None:
    """Every band spelling lands as the same band COUNT/STRATEGY in the plan.

    ``describe_plan`` is the introspection seam a runner logs, so the resolved
    merge shape must be readable from it: the band count, one size per band summing
    to the function count, the strategy that produced them, the ``switch_ops`` flag
    and the alignment.  The numbers are ``test_batch_engine.py``'s contract.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), **knobs)
    info = _info(plan)
    assert info["band_count"] == expected_bands, (knobs, info["band_count"])
    assert info["band_strategy"] == expected_strategy, (knobs, info["band_strategy"])
    assert len(info["band_sizes"]) == info["band_count"], info["band_sizes"]
    assert sum(info["band_sizes"]) == info["n_functions"] == POP_SIZE
    assert info["functions"] == POP_SIZE, "the four programs are distinct"
    assert info["switch_ops"] is bool(knobs.get("switch_ops", False))
    assert info["alignment"] == B.DEFAULT_ALIGNMENT
    assert info["variant"] == B.LOSS_VARIANT and info["per_row"] is False


#: More distinct programs than ``DEFAULT_BAND_SIZE`` (32) -- the only population
#: shape on which the flat spellings and the band-SIZE model disagree, i.e. the
#: shape that tells "the flat merge was selected" from "the model happened to ask
#: for one band".  40 distinct programs put the default model at
#: ``ceil(40 / 32) == 2`` bands.
WIDE_POP_SIZE = 40
WIDE_BAND_COUNT = -(-WIDE_POP_SIZE // B.DEFAULT_BAND_SIZE)

#: The spellings that must all select the single flat band, ``bands=1`` included.
_FLAT_SPELLINGS = (
    pytest.param({"bands": 1}, id="bands-1"),
    pytest.param({"bands": "none"}, id="bands-none"),
    pytest.param({"bands": "flat"}, id="bands-flat"),
    pytest.param({"bands": "0"}, id="bands-zero"),
    pytest.param({"bands": 0}, id="bands-int-zero"),
)


def _wide_population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``WIDE_POP_SIZE`` pairwise-distinct programs: ``x0 + <distinct constant>``.

    Distinct constants keep every row a separate merged function (nothing dedups),
    which is what puts the band-size model above one band.
    """
    specs = [
        te.fn(te.ADD, te.var(0), te.const(0.5 + index))
        for index in range(WIDE_POP_SIZE)
    ]
    return te.rows_from_specs(specs, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


@pytest.mark.parametrize("knobs", _FLAT_SPELLINGS)
def test_a_flat_spelling_selects_the_flat_merge_over_the_band_size_model(
    tiny_pop: None, knobs: Dict[str, Any]
) -> None:
    """On a bandable population every flat spelling collapses the merge to ONE band.

    ``test_the_band_knobs_reach_the_plan`` cannot see this: its four programs make
    the size model answer ``K == 1`` too, so a flat spelling that fell through to
    the model looked exactly like a flat merge (that is how the ``None`` regression
    stayed green while measuring ``K == 8`` on a 231-function population).  Here
    the model really bands (``WIDE_BAND_COUNT``), so each flat spelling -- and
    ``bands=1`` -- has to return ``band_count == 1``, the ``"flat"`` strategy, one
    band of all ``n_functions`` functions and the SAME emitted ops.
    The cache is cleared between the plans: ``bands=1`` and ``bands="flat"``
    resolve to the same memo key, so without it the later spelling would be
    answered from the earlier plan instead of re-planning.
    """
    values, types, sizes = _wide_population()
    x = _x()

    model = _info(_plan(values, types, sizes, x))
    assert model["functions"] == model["n_functions"] == WIDE_POP_SIZE
    assert WIDE_BAND_COUNT >= 2
    assert model["band_count"] == WIDE_BAND_COUNT, model["band_sizes"]
    assert model["band_strategy"] == "explicit", model["band_strategy"]
    assert len(model["band_sizes"]) == WIDE_BAND_COUNT
    assert sum(model["band_sizes"]) == WIDE_POP_SIZE

    E.clear_cache()
    reference = _info(_plan(values, types, sizes, x, bands=1))
    E.clear_cache()
    flat = _info(_plan(values, types, sizes, x, **knobs))
    assert flat["band_count"] == 1, (knobs, flat["band_sizes"])
    assert flat["band_strategy"] == "flat", (knobs, flat["band_strategy"])
    assert flat["band_sizes"] == [WIDE_POP_SIZE], (knobs, flat["band_sizes"])
    assert flat["emitted_ops"] == reference["emitted_ops"]
    assert flat["emitted_ops"] != model["emitted_ops"], (
        "the banded and the flat merge must not emit the same graph, otherwise "
        "this fixture could not tell them apart"
    )
    assert flat == reference


@pytest.mark.parametrize("mode", list(B.ALIGNMENTS))
def test_the_alignment_argument_reaches_the_plan(tiny_pop: None, mode: str) -> None:
    """The alignment mode is recorded in the plan, in its canonical spelling.

    It is a heuristic, so nothing may silently normalize it away: a logged plan has
    to state which alignment the merged class DAGs were built with.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), alignment=mode.upper())
    info = _info(plan)
    assert info["alignment"] == mode


def test_the_environment_knobs_produce_the_same_plan_as_the_arguments(
    tiny_pop: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EVOGP_BATCH_BANDS`` / ``EVOGP_BATCH_BAND_SIZE`` equal their arguments.

    The plan memo is keyed on the RESOLVED knob values, so an env-driven call could
    hide a divergence behind a hit; the cache is therefore cleared between the two
    calls and the plans are compared field by field through ``describe_plan``.
    """
    values, types, sizes = _population()
    x = _x()

    explicit_bands = _info(_plan(values, types, sizes, x, bands=2))
    E.clear_cache()
    monkeypatch.setenv(B.BANDS_ENV, "2")
    from_env_bands = _info(_plan(values, types, sizes, x))
    assert from_env_bands == explicit_bands, from_env_bands

    monkeypatch.delenv(B.BANDS_ENV)
    explicit_size = _info(_plan(values, types, sizes, x, band_size=1))
    E.clear_cache()
    monkeypatch.setenv(B.BAND_SIZE_ENV, "1")
    from_env_size = _info(_plan(values, types, sizes, x))
    assert from_env_size == explicit_size, from_env_size
    assert from_env_size["band_count"] == POP_SIZE


# =========================================================================== #
# 9. the front tables: identity semantics and counters
# =========================================================================== #
def test_a_first_call_misses_then_a_repeat_call_hits_the_plan(tiny_pop: None) -> None:
    """The first call pays both tables; the second is a plan hit and re-compiles not.

    A plan hit returns the stored entry BEFORE ``build_batch`` is consulted, so the
    ``batch_exec_*`` counters stay untouched on the repeat -- the documented
    short-circuit, and the reason a repeat call is free.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    after_first = B.batch_cache_stats()
    assert after_first["batch_plan_misses"] == 1
    assert after_first["batch_exec_misses"] == 1
    assert after_first["batch_plan_entries"] == 1
    assert after_first["batch_exec_entries"] == 1
    second = _loss(values, types, sizes, inputs, labels)
    after_second = B.batch_cache_stats()
    assert after_second["batch_plan_hits"] == after_first["batch_plan_hits"] + 1
    assert after_second["batch_plan_misses"] == 1, "the second call must not re-plan"
    assert after_second["batch_exec_misses"] == 1, "the second call must not re-compile"
    assert after_second["batch_exec_hits"] == after_first["batch_exec_hits"], (
        "a plan hit returns before build_batch is consulted"
    )
    assert after_second["batch_plan_entries"] == 1
    assert after_second["batch_exec_entries"] == 1
    assert torch.equal(first, second)


def test_a_repeat_call_skips_the_host_plan(
    tiny_pop: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan hit skips the decode, the traces and the merge -- and the build.

    Counters (never wall-clock) prove it: ``_plan_host`` is the whole host planning
    step (``decode_population`` + dedup + one trace per program + the merge), and
    ``build_batch`` is the only compile site.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    plans = _Counter()
    plans.install(monkeypatch, B, "_plan_host")
    builds = _Counter()
    builds.install(monkeypatch, B, "build_batch")
    _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1 and builds.calls == 1
    _loss(values, types, sizes, inputs, labels)
    _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1, "a plan hit must not re-decode / re-trace / re-merge"
    assert builds.calls == 1, "a plan hit never reaches build_batch"


def test_a_rebuilt_equal_population_hits_the_content_level_of_the_plan_key(
    tiny_pop: None,
) -> None:
    """Same content, new buffers: an identity miss but a CONTENT hit.

    The plan key is two-level.  Its first level is still IDENTITY
    (``data_ptr``/shape/dtype/device/``_version``), so a copy is a
    ``batch_plan_*`` miss; its second level is a digest over the decoded programs,
    so an EQUAL copy hits -- ``batch_content_hits`` grows, the host work (decode +
    dedup + trace + merge) and the compile are NOT re-paid, and the re-bound plan
    answers with the identical loss.  ``test_batch_content_key.py`` owns the rest of
    that key (mutations, collisions, invalidation); this pins the identity/content
    split on the knob file's own fixture.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    copied = _population()
    assert torch.equal(values, copied[0]), "the fixture copy must have the same content"
    second = _loss(*copied, inputs, labels)
    stats = B.batch_cache_stats()
    assert stats["batch_plan_misses"] == 2, "a copied population is a new identity token"
    assert stats["batch_plan_hits"] == 0, "the identity level cannot serve a new buffer"
    assert stats["batch_exec_misses"] == 1, "the merged multiset is unchanged"
    assert stats["batch_exec_hits"] == 0, (
        "a content hit returns before build_batch is consulted"
    )
    assert stats["batch_exec_entries"] == 1
    content = B.batch_content_stats()
    assert content["batch_content_hits"] == 1, content
    assert content["batch_content_misses"] == 1, "the first call still paid the digest"
    assert content["batch_content_entries"] == 1
    assert torch.equal(first, second), "the content-level hit serves the same numbers"


def test_clear_cache_resets_both_batch_tables_and_repays_the_work(tiny_pop: None) -> None:
    """``evoxc_engine.clear_cache()`` is the single invalidation point of both tables.

    It zeroes every batch counter AND drops the entries (through the registered
    hook), so the next call re-pays the plan and the compile -- the engine's
    documented "clear the cache after changing a compile knob" contract.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    assert B.batch_cache_stats()["batch_plan_entries"] == 1
    E.clear_cache()
    cleared = B.batch_cache_stats()
    for table, memo in (
        ("batch_plan", B.BATCH_PLAN_MEMO),
        ("batch_exec", B.BATCH_EXEC_MEMO),
    ):
        assert cleared[f"{table}_hits"] == 0, table
        assert cleared[f"{table}_misses"] == 0, table
        assert cleared[f"{table}_entries"] == 0, table
        assert cleared[f"{table}_maxsize"] == memo.maxsize, table
    again = _loss(values, types, sizes, inputs, labels)
    after = B.batch_cache_stats()
    assert after["batch_plan_misses"] == 1, "clear_cache() must re-pay the plan work"
    assert after["batch_exec_misses"] == 1, "and the compile"
    assert after["batch_plan_hits"] == 0 and after["batch_exec_hits"] == 0
    assert torch.equal(first, again)


def test_an_all_constant_population_reuses_one_executable_across_pop_sizes(
    tiny_pop: None,
) -> None:
    """One merged function: pop 4 -> 64 -> 256 is one compile and two exec hits.

    Every row of an all-constant population collapses onto the same merged
    function, so the compiled body is identical at all three sizes while the row
    expansion is not.  The counters pin the reuse (the plan is still a per-population
    identity, so it misses each time) and the loss rows pin the re-binding.
    """
    inputs, labels = _x(4, seed=7), _labels(4, seed=9)
    expected = float(((3.5 - labels[:, 0]) ** 2).sum() / labels.shape[0])
    for pop_size in (4, 64, 256):
        loss = _loss(*_constant_rows(pop_size), inputs, labels)
        assert tuple(loss.shape) == (pop_size,), (
            f"pop {pop_size}: the loss came back as {tuple(loss.shape)}"
        )
        assert torch.equal(loss, loss[0].expand_as(loss)), (
            f"pop {pop_size}: the constant loss must be the same on every row"
        )
        assert abs(float(loss[0]) - expected) < 1e-6
    stats = B.batch_cache_stats()
    assert stats["batch_exec_misses"] == 1, "one merged function -> exactly one compile"
    assert stats["batch_exec_hits"] == 2, "the same multiset at another size is a hit"
    assert stats["batch_exec_entries"] == 1
    assert stats["batch_plan_misses"] == 3, "three populations are three plan keys"
    assert stats["batch_plan_hits"] == 0


def test_the_batch_tables_are_disjoint_from_every_other_stats_surface() -> None:
    """Five stats surfaces, five disjoint key sets.

    ``cache_stats()`` (exactly its 8 keys) and ``dense_cache_stats()`` are pinned by
    other files in this suite, and the disk layer by a third: the batch tables
    registered a plain reset HOOK precisely so they never appear in the merged
    ``dense_memo`` stats.  The CONTENT table of the two-level plan key
    (``batch_content_stats()``, all ``batch_content_*``) is the fifth surface and is
    disjoint from all of them as well -- a ``batch_content_*`` key in the cache
    counters (or a cache counter in the content table) is a regression.
    """
    batch_keys = set(B.batch_cache_stats())
    dense_keys = set(D.dense_cache_stats())
    engine_keys = set(E.cache_stats())
    artifact_keys = set(E.artifact_cache_stats())
    content_keys = set(B.batch_content_stats())
    assert batch_keys == set(_BATCH_STAT_KEYS)
    assert content_keys == set(_CONTENT_STAT_KEYS), sorted(content_keys)
    assert engine_keys == set(_ENGINE_STAT_KEYS), sorted(engine_keys)
    assert dense_keys, "the dense route registers its own tables"
    assert batch_keys.isdisjoint(dense_keys), sorted(batch_keys & dense_keys)
    assert batch_keys.isdisjoint(engine_keys), sorted(batch_keys & engine_keys)
    assert batch_keys.isdisjoint(artifact_keys), sorted(batch_keys & artifact_keys)
    assert batch_keys.isdisjoint(content_keys), sorted(batch_keys & content_keys)
    for name, keys in (
        ("dense_cache_stats", dense_keys),
        ("cache_stats", engine_keys),
        ("artifact_cache_stats", artifact_keys),
        ("batch_content_stats", content_keys),
    ):
        assert content_keys.isdisjoint(keys), (name, sorted(content_keys & keys))
    leaked = sorted(
        key
        for key in dense_keys | engine_keys | artifact_keys | content_keys
        if key.startswith("batch_")
    )
    assert not leaked, f"a batch table leaked into another stats surface: {leaked}"
    content_leaked = sorted(
        key for key in batch_keys if key.startswith("batch_content_")
    )
    assert not content_leaked, (
        f"the content table leaked into batch_cache_stats(): {content_leaked}"
    )


# =========================================================================== #
# 10. declines: BatchUnsupported vs ValueError
# =========================================================================== #
def test_batch_unsupported_is_a_declining_runtime_error() -> None:
    """A decline is a ``RuntimeError`` the call sites can catch and fall back on.

    It must NOT be a ``ValueError``: a caller distinguishes "not my population"
    (fall back to the specialized engine) from "your configuration is malformed"
    (fail loudly), so the two hierarchies stay separate.
    """
    assert issubclass(B.BatchUnsupported, RuntimeError)
    assert not issubclass(B.BatchUnsupported, ValueError)


def test_the_population_threshold_declines_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pop_size < EVOGP_BATCH_MIN_POP`` declines with the variable and both numbers.

    The default (64) declines the 4-row fixture, an explicit lower threshold (8)
    declines it too and names ITS value, and raising the threshold down to 1 admits
    it -- the three states the wiring task has to tell apart.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    with pytest.raises(B.BatchUnsupported) as excinfo:
        _loss(values, types, sizes, inputs, labels)
    message = str(excinfo.value)
    assert B.MIN_POP_ENV in message, message
    assert str(B.DEFAULT_MIN_POP) in message and str(POP_SIZE) in message, message
    monkeypatch.setenv(B.MIN_POP_ENV, "8")
    with pytest.raises(B.BatchUnsupported) as excinfo:
        _loss(values, types, sizes, inputs, labels)
    message = str(excinfo.value)
    assert B.MIN_POP_ENV in message and "8" in message, message
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    assert tuple(_loss(values, types, sizes, inputs, labels).shape) == (POP_SIZE,)


@pytest.mark.parametrize("entry", ["loss", "value"])
def test_an_empty_population_declines(tiny_pop: None, entry: str) -> None:
    """An empty population is a decline with its own message, not a malformed one.

    It must not be reported as merely "too small": nothing can be merged, and the
    caller has nothing to fall back on either -- but it is still this route's
    "not mine" answer.
    """
    empty = (
        torch.zeros((0, ROW_LEN), dtype=torch.float32),
        torch.zeros((0, ROW_LEN), dtype=torch.int16),
        torch.zeros((0, ROW_LEN), dtype=torch.int16),
    )
    with pytest.raises(B.BatchUnsupported) as excinfo:
        if entry == "loss":
            _loss(*empty, _x(), _labels())
        else:
            _value(*empty, _x(), per_row=False)
    assert "empty population" in str(excinfo.value), str(excinfo.value)


def test_an_unknown_backend_name_is_a_decline(tiny_pop: None) -> None:
    """An unknown backend NAME is a decline, not the resolver's ``ValueError``.

    "This route does not build that" is a fallback answer, so all three entry
    points convert it; a malformed *option* of a supported backend would stay a
    ``ValueError``.  ``numpy`` / ``iree`` are the only names it builds.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    with pytest.raises(B.BatchUnsupported) as excinfo:
        B.plan_batch(
            values, types, sizes, INPUT_LEN, 1, inputs, per_row=False, backend="tpu"
        )
    message = str(excinfo.value)
    assert "tpu" in message and "numpy" in message, message
    with pytest.raises(B.BatchUnsupported):
        B.sr_fitness_batch(
            values, types, sizes, INPUT_LEN, 1, inputs, labels, True, backend="tpu"
        )
    with pytest.raises(B.BatchUnsupported):
        B.evaluate_batch(
            values, types, sizes, INPUT_LEN, 1, inputs, per_row=False, backend="tpu"
        )


@pytest.mark.parametrize(
    "case",
    ["per_row_row_count", "input_len_columns", "x_rank", "label_shape"],
)
def test_input_shape_doubts_decline(tiny_pop: None, case: str) -> None:
    """Every ``x`` / labels shape doubt is a decline, never a ``ValueError``.

    The fallback engine owns the error contract for a malformed call, so a route
    that declares "this is not mine" must not pre-empt it: the message names the
    mismatch, and the caller retries on the specialized engine.
    """
    values, types, sizes = _population()
    if case == "per_row_row_count":
        with pytest.raises(B.BatchUnsupported) as excinfo:
            _value(values, types, sizes, _x(POP_SIZE + 1), per_row=True)
        assert "per_row" in str(excinfo.value), str(excinfo.value)
    elif case == "input_len_columns":
        with pytest.raises(B.BatchUnsupported) as excinfo:
            _value(values, types, sizes, _x()[:, : INPUT_LEN - 1], per_row=False)
        assert "input_len" in str(excinfo.value), str(excinfo.value)
    elif case == "x_rank":
        with pytest.raises(B.BatchUnsupported) as excinfo:
            _value(values, types, sizes, _x().reshape(-1), per_row=False)
        assert "2-D" in str(excinfo.value), str(excinfo.value)
    else:
        with pytest.raises(B.BatchUnsupported) as excinfo:
            _loss(values, types, sizes, _x(), _labels(N_ROWS + 1))
        assert "labels" in str(excinfo.value), str(excinfo.value)


def test_malformed_knobs_stay_value_errors(tiny_pop: None) -> None:
    """A bad knob or an unknown variant is a ``ValueError`` from every entry point.

    These are configuration mistakes, not "not my population": the message names the
    offending knob (or variant) so the caller can fix it instead of falling back,
    and the distinction from :class:`BatchUnsupported` is asserted here.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    with pytest.raises(ValueError) as excinfo:
        _loss(values, types, sizes, inputs, labels, bands="maybe")
    assert B.BANDS_ENV in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _plan(values, types, sizes, inputs, band_size=0)
    assert B.BAND_SIZE_ENV in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _plan(values, types, sizes, inputs, alignment="nope")
    assert B.ALIGNMENT_ENV in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        B.plan_batch(
            values,
            types,
            sizes,
            INPUT_LEN,
            1,
            inputs,
            per_row=False,
            variant="nope",
            backend=BACKEND,
        )
    message = str(excinfo.value)
    assert "nope" in message and B.VALUE_VARIANT in message, message
