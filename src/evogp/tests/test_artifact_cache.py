"""Regression tests for the opt-in on-disk executable artifact layer.

What this file pins
-------------------
``evogp.tree.artifact_cache`` -- the disk layer both evaluator routes consult on an
in-process memo miss (``evoxc_engine._executable`` for the specialized route,
``dense_engine.build_dense`` for the dense one), switched on by
``EVOGP_EVOXC_ARTIFACT_DIR``. The properties a cache of *compiled* code must have,
and the ones a wrong implementation silently breaks:

* **off by default** -- with the switch unset, one environment read is the whole
  cost: no key is derived, no directory is created, no file is opened, and the
  routes build exactly as before;
* **a stable, process-independent key** -- derived from the graph name, the build's
  spec tuple (shapes AND static content), the backend, the device and the frozen
  options (never ``id(fn)``), so two equally-built plans in the same process (and
  two processes) agree, while a different graph/backend/device/option set does not;
* **a corrupt or foreign artifact can only cost a rebuild** -- a truncated file, a
  file whose signature does not fit the requested call shape, or a file for another
  plan is discarded and the real ``build_cached`` path runs;
* **saving is best effort** -- a backend executable without a usable ``save``
  (the ``PersistenceError`` case) leaves no file and never raises into the caller;
* **two separate lifetimes** -- ``evoxc_engine.clear_cache()`` resets the in-memory
  tables and leaves the disk alone, ``evoxc_engine.clear_artifacts()`` deletes the
  files and leaves memory alone, and ``artifact_cache_stats()`` shares no key with
  ``cache_stats()`` / ``dense_cache_stats()``.

Hermeticity
-----------
Everything here runs on the ``numpy`` backend (CPU interpreter, no device probe, no
iree compile, no wall-clock assertion -- counts and file contents only), on tiny
hand-built populations (``P=4``...``P=8``, ``L=8``, ``N=12``), with the artifact
directory pointed at pytest's ``tmp_path`` through ``monkeypatch`` so the switch is
never left on for another test. The in-process tables are process-global, so every
test that counts calls calls ``evoxc_engine.clear_cache()`` first and asserts
deltas it established itself.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python -m pytest \\
        src/evogp/tests/test_artifact_cache.py -q -p no:cacheprovider
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from typing import Any, Iterator, List, Tuple

import pytest
import torch

import evoxc
from evogp.tree import Forest
from evogp.tree import artifact_cache as A
from evogp.tree import dense_engine as D
from evogp.tree import evoxc_engine as E

# --------------------------------------------------------------------------- #
# the suite's tree-spec DSL (var/const/fn -> prefix rows) is reused, not copied
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import test_equivalence as _eq
except ImportError:  # pragma: no cover - only under pytest's importlib import mode
    _spec = importlib.util.spec_from_file_location(
        "evogp_artifact_helpers", _HERE / "test_equivalence.py"
    )
    _eq = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eq)

# --------------------------------------------------------------------------- #
# fixtures: tiny CPU population + data (no device, no generation, no CUDA)
# --------------------------------------------------------------------------- #

INPUT_LEN = 3
POP = 4
ROW_LEN = 8
N_ROWS = 12


def population() -> Forest:
    """4 hand-built trees (``x0``, ``x0+x1``, ``(x1*x2)-0.5``, ``x2``), CPU tensors."""
    specs = [
        _eq.var(0),
        _eq.fn(1, _eq.var(0), _eq.var(1)),  # ADD
        _eq.fn(2, _eq.fn(3, _eq.var(1), _eq.var(2)), _eq.const(0.5)),  # SUB(MUL, 0.5)
        _eq.var(2),
    ]
    values, types, sizes = _eq.rows_from_specs(specs, 1, pad_to=ROW_LEN)
    return Forest(INPUT_LEN, 1, values, types, sizes)


def data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic ``(N, input_len)`` inputs and ``(N, 1)`` labels on the CPU."""
    gen = torch.Generator().manual_seed(11)
    x = torch.rand(N_ROWS, INPUT_LEN, generator=gen, dtype=torch.float32)
    y = torch.rand(N_ROWS, 1, generator=gen, dtype=torch.float32)
    return x, y


def dense_loss(forest: Forest, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The dense route's SR loss at numpy backend, explicit in every knob."""
    return D.sr_fitness_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        x,
        y,
        True,
        backend="numpy",
    )


def specialized_loss(
    forest: Forest, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """The specialized (per-structure) route's SR loss at numpy backend."""
    population_ = E.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    return E.sr_fitness(population_, x, y, True, backend="numpy")


@pytest.fixture(autouse=True)
def clean_tables() -> Iterator[None]:
    """Start and end every test from empty in-memory tables (process-global)."""
    E.clear_cache()
    yield
    E.clear_cache()


@pytest.fixture
def store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Enable the disk layer on a fresh directory and reset its counters."""
    directory = tmp_path / "artifacts"
    monkeypatch.setenv(A.DIR_ENV, str(directory))
    A.reset_stats()
    return directory


@pytest.fixture
def disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default state: the switch is off."""
    monkeypatch.delenv(A.DIR_ENV, raising=False)
    A.reset_stats()


def _files(directory: pathlib.Path) -> List[pathlib.Path]:
    """The artifact files in ``directory`` (empty when it does not exist)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"*{A.SUFFIX}"))


# =========================================================================== #
# 1. off by default: nothing on disk, nothing derived
# =========================================================================== #


def test_unset_directory_is_the_off_state(
    disabled: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch off means ``artifact_path`` says "do nothing" without any I/O."""
    specs = (evoxc.TensorSpec((4, 3), evoxc.float32), 3)
    device = evoxc.core.Device("cpu", 0)
    assert A.artifact_dir() is None
    assert A.artifact_path("dense:loss", specs, "numpy", device, ()) is None
    # ... and it never even attempts to derive a key (which is where the hashing is)
    monkeypatch.setattr(
        A, "compute_key", lambda *_: pytest.fail("a key was derived while disabled")
    )
    monkeypatch.setattr(
        A, "artifact_key", lambda *a, **k: pytest.fail("artifact_key called while off")
    )
    assert A.artifact_path("dense:loss", specs, "numpy", device, ()) is None
    assert A.stats()["artifact_enabled"] == 0
    assert A.stats()["artifact_files"] == 0


def test_off_route_touches_no_filesystem(
    disabled: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_dense`` / ``_executable`` with the switch off do not load or save."""
    monkeypatch.setattr(
        A, "load", lambda *a, **k: pytest.fail("artifact load attempted while off")
    )
    monkeypatch.setattr(
        A, "save", lambda *a, **k: pytest.fail("artifact save attempted while off")
    )
    forest, (x, y) = population(), data()
    assert torch.allclose(dense_loss(forest, x, y), specialized_loss(forest, x, y),
                          rtol=1e-6, atol=1e-6)
    assert _files(tmp_path / "artifacts") == []
    assert not (tmp_path / "artifacts").exists()


# =========================================================================== #
# 2. the key: stable, value-typed, discriminating
# =========================================================================== #


def test_key_is_stable_across_equal_plans_and_graph_identities(
    disabled: None,
) -> None:
    """Equal specs -> equal key, whatever object identity produced them.

    The point of the key is that it never contains ``id(fn)``: two independently
    built (but equal) spec tuples, and two names for the same graph body, must land
    on the same file in the same process and in the next one.
    """
    device = evoxc.core.Device("cpu", 0)
    left = (evoxc.TensorSpec((8, 4), evoxc.float32), 4)
    right = (evoxc.TensorSpec((8, 4), evoxc.float32), 4)  # separate objects
    assert left is not right and left[0] is not right[0]
    first = A.artifact_key("specialized", left, "numpy", device, ())
    second = A.artifact_key("specialized", right, "numpy", device, ())
    assert first == second
    assert A.artifact_key("specialized", left, "numpy", device, ()) == first
    # the readable prefix, then one sha256 hex digest
    prefix, _, digest = first.partition("-")
    assert prefix == "specialized"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_key_separates_everything_that_selects_the_compiled_body(
    disabled: None,
) -> None:
    """Name, specs (incl. static content), backend, device and options all differ."""
    device = evoxc.core.Device("cpu", 0)
    base = (evoxc.TensorSpec((8, 4), evoxc.float32), 4)
    key = A.artifact_key("dense:loss", base, "numpy", device, ())
    others = {
        "graph name": A.artifact_key("dense:value", base, "numpy", device, ()),
        "tensor shape": A.artifact_key(
            "dense:loss", (evoxc.TensorSpec((8, 5), evoxc.float32), 4), "numpy", device, ()
        ),
        "tensor dtype": A.artifact_key(
            "dense:loss", (evoxc.TensorSpec((8, 4), evoxc.int32), 4), "numpy", device, ()
        ),
        "static content": A.artifact_key(
            "dense:loss", (evoxc.TensorSpec((8, 4), evoxc.float32), 5), "numpy", device, ()
        ),
        "backend": A.artifact_key("dense:loss", base, "iree", device, ()),
        "device kind": A.artifact_key(
            "dense:loss", base, "numpy", evoxc.core.Device("rocm", 0), ()
        ),
        "device index": A.artifact_key(
            "dense:loss", base, "numpy", evoxc.core.Device("rocm", 1), ()
        ),
        "options": A.artifact_key("dense:loss", base, "numpy", device, (("opt_level", "O0"),)),
    }
    for label, other in others.items():
        assert other != key, f"the key does not separate {label}"


def test_key_follows_the_real_plan_parameters(store: pathlib.Path) -> None:
    """Two plans of the same shape share a path; a different shape does not."""
    forest = population()
    plan_a = D.plan_dense(
        forest.batch_node_value, forest.batch_node_type, forest.batch_subtree_size,
        forest.input_len, forest.output_len, N_ROWS, per_row=False, backend="numpy",
        variant=D.LOSS_VARIANT,
    )
    plan_b = D.plan_dense(
        forest.batch_node_value, forest.batch_node_type, forest.batch_subtree_size,
        forest.input_len, forest.output_len, N_ROWS, per_row=False, backend="numpy",
        variant=D.LOSS_VARIANT,
    )
    plan_c = D.plan_dense(
        forest.batch_node_value, forest.batch_node_type, forest.batch_subtree_size,
        forest.input_len, forest.output_len, N_ROWS + 1, per_row=False, backend="numpy",
        variant=D.LOSS_VARIANT,
    )
    device = evoxc.core.Device("cpu", 0)
    name = D.DENSE_ARTIFACT_NAMES[D.LOSS_VARIANT]
    path_a = A.artifact_path(name, plan_a.specs, "numpy", device, ())
    path_b = A.artifact_path(name, plan_b.specs, "numpy", device, ())
    path_c = A.artifact_path(name, plan_c.specs, "numpy", device, ())
    assert path_a is not None and path_a == path_b
    assert path_c != path_a
    assert name == "dense:loss"


def test_unencodable_spec_disables_the_layer_for_that_build(
    store: pathlib.Path,
) -> None:
    """A spec the key codec cannot encode yields "no artifact", never an error."""
    assert store.is_dir() is False
    path = A.artifact_path("dense:loss", (object(),), "numpy", evoxc.core.Device("cpu", 0), ())
    assert path is None


# =========================================================================== #
# 3. round trip on both routes (numpy): store -> hit, same numbers
# =========================================================================== #


@pytest.mark.parametrize("route", ["dense", "specialized"])
def test_round_trip_stores_then_hits_and_keeps_the_numbers(
    route: str, store: pathlib.Path
) -> None:
    """First call saves the artifact; after ``clear_cache`` the same call loads it.

    Both routes go through the disk layer: the dense one on its single per-shape
    graph, the specialized one per distinct tree structure (this population's four
    trees are pairwise distinct programs, so four files + the dense one).
    """
    forest, (x, y) = population(), data()
    evaluate = dense_loss if route == "dense" else specialized_loss

    first = evaluate(forest, x, y)
    files = _files(store)
    expected = 1 if route == "dense" else 4
    assert len(files) == expected, f"{[f.name for f in files]}"
    assert all(f.stat().st_size > 0 for f in files)
    stats = A.stats()
    assert stats["artifact_stores"] == expected
    assert stats["artifact_misses"] == expected

    E.clear_cache()  # forget every in-memory table, keep the files
    assert len(_files(store)) == expected
    again = evaluate(forest, x, y)
    stats = A.stats()
    assert stats["artifact_hits"] == expected, stats
    assert stats["artifact_misses"] == expected  # no new misses: nothing was rebuilt
    assert stats["artifact_load_failures"] == 0
    assert torch.allclose(first, again, rtol=0.0, atol=0.0), "the hit changed the loss"

    # ... and the loaded executable is the same program: the routes still agree
    reference = (
        specialized_loss(forest, x, y) if route == "dense" else dense_loss(forest, x, y)
    )
    torch.testing.assert_close(again, reference, rtol=1e-6, atol=1e-6)


def test_loaded_artifact_is_numerically_the_same_program(store: pathlib.Path) -> None:
    """The loss of a loaded artifact equals the loss of the freshly built graph."""
    forest, (x, y) = population(), data()
    built = dense_loss(forest, x, y)
    E.clear_cache()
    loaded = dense_loss(forest, x, y)
    assert A.stats()["artifact_hits"] == 1
    assert torch.equal(built, loaded)


# =========================================================================== #
# 4. corrupt / foreign artifacts fall back to a build
# =========================================================================== #


def test_truncated_artifact_is_discarded_and_rebuilt(store: pathlib.Path) -> None:
    """A torn file is a rebuild (and is rewritten whole), never a wrong number."""
    forest, (x, y) = population(), data()
    expected = dense_loss(forest, x, y)
    (artifact,) = _files(store)
    good_size = artifact.stat().st_size
    with open(artifact, "r+b") as handle:
        handle.truncate(good_size // 3)

    E.clear_cache()
    rebuilt = dense_loss(forest, x, y)
    stats = A.stats()
    assert stats["artifact_load_failures"] == 1, stats
    assert stats["artifact_stores"] == 2
    assert torch.equal(rebuilt, expected)
    (artifact,) = _files(store)
    assert artifact.stat().st_size == good_size


def test_foreign_artifact_is_rejected_by_the_signature_check(store: pathlib.Path) -> None:
    """An artifact whose signature does not fit the call shape is rebuilt."""
    forest, (x, y) = population(), data()
    expected = dense_loss(forest, x, y)
    (artifact,) = _files(store)
    specs = (
        evoxc.TensorSpec((ROW_LEN, POP), evoxc.int32),  # right shape, wrong dtype
    )
    assert A.load(str(artifact), specs, "numpy", evoxc.core.Device("cpu", 0)) is None
    assert not artifact.exists(), "an unusable artifact must be deleted"
    assert A.stats()["artifact_load_failures"] == 1

    E.clear_cache()
    assert torch.equal(dense_loss(forest, x, y), expected)


def test_signature_matches_checks_shapes_dtypes_and_statics(store: pathlib.Path) -> None:
    """The validator: real executable + its own specs pass, every change fails."""
    forest, (x, y) = population(), data()
    plan = D.plan_dense(
        forest.batch_node_value, forest.batch_node_type, forest.batch_subtree_size,
        forest.input_len, forest.output_len, N_ROWS, per_row=False, backend="numpy",
        variant=D.LOSS_VARIANT,
    )
    executable = D.build_dense(plan, backend="numpy")
    assert A.signature_matches(executable, plan.specs) is True

    tensors = [spec for spec in plan.specs if isinstance(spec, evoxc.TensorSpec)]
    statics = plan.specs[-1]
    wrong_shape = (evoxc.TensorSpec((ROW_LEN, POP + 1), evoxc.int32),) + tuple(tensors[1:]) + (statics,)
    wrong_dtype = (evoxc.TensorSpec((ROW_LEN, POP), evoxc.float32),) + tuple(tensors[1:]) + (statics,)
    wrong_static = tuple(tensors) + (tuple(statics[:-1]) + (not statics[-1],),)
    missing = tuple(tensors[:-1]) + (statics,)
    for label, specs in (
        ("shape", wrong_shape),
        ("dtype", wrong_dtype),
        ("static", wrong_static),
        ("arity", missing),
    ):
        assert A.signature_matches(executable, specs) is False, label
    assert A.signature_matches(object(), plan.specs) is False


# =========================================================================== #
# 5. saving is best effort
# =========================================================================== #


class _NoSave:
    """An executable stand-in whose backend cannot serialize (evoxc raises)."""

    def save(self, path: Any) -> None:
        raise evoxc.core.PersistenceError("this backend has no CompiledArtifact")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "_NoSave()"


def test_save_failure_is_swallowed_and_leaves_no_file(store: pathlib.Path) -> None:
    """A backend without ``save`` -> ``False``, one counter, no file, no exception."""
    target = str(store / ("artifact" + A.SUFFIX))
    assert A.save(_NoSave(), target) is False
    assert not pathlib.Path(target).exists()
    assert list(pathlib.Path(target).parent.glob("*.tmp-*")) == []
    stats = A.stats()
    assert stats["artifact_save_failures"] == 1
    assert stats["artifact_stores"] == 0
    assert stats["artifact_files"] == 0


def test_save_creates_the_directory_on_demand(store: pathlib.Path) -> None:
    """The directory appears on the first successful save, not before."""
    class _Saves:
        def save(self, path: Any) -> None:
            pathlib.Path(path).write_bytes(b"payload")

    assert store.is_dir() is False
    target = store / ("graph" + A.SUFFIX)
    assert A.save(_Saves(), str(target)) is True
    assert store.is_dir() is True
    assert target.read_bytes() == b"payload"
    assert A.stats()["artifact_files"] == 1
    assert A.stats()["artifact_stores"] == 1


def test_missing_file_is_a_plain_miss(store: pathlib.Path) -> None:
    """No file -> ``None`` and a miss counter; nothing is created."""
    path = str(store / ("absent" + A.SUFFIX))
    assert A.load(path, (), "numpy", evoxc.core.Device("cpu", 0)) is None
    stats = A.stats()
    assert stats["artifact_misses"] == 1
    assert stats["artifact_load_failures"] == 0
    assert stats["artifact_files"] == 0


# =========================================================================== #
# 6. two lifetimes: memory vs disk
# =========================================================================== #


def test_clear_cache_keeps_artifacts_and_clear_artifacts_keeps_memory(
    store: pathlib.Path,
) -> None:
    """``clear_cache()`` is in-memory only; ``clear_artifacts()`` is disk only."""
    forest, (x, y) = population(), data()
    dense_loss(forest, x, y)
    (artifact,) = _files(store)
    memory = dict(E.cache_stats())

    E.clear_cache()
    assert artifact.exists(), "clear_cache() must not delete artifacts"
    assert set(memory) == set(E.cache_stats())
    assert E.cache_stats()["memo_entries"] == 0
    assert A.stats()["artifact_stores"] == 1

    dense_loss(forest, x, y)  # refill the in-memory tables from the artifact
    assert D.dense_cache_stats()["dense_exec_entries"] > 0
    filled = D.dense_cache_stats()

    E.clear_artifacts()
    assert not artifact.exists(), "clear_artifacts() must delete the artifacts"
    assert D.dense_cache_stats() == filled, "clear_artifacts() must not touch memory"
    stats = E.artifact_cache_stats()
    assert stats["artifact_stores"] == 0 and stats["artifact_hits"] == 0
    assert stats["artifact_files"] == 0


def test_artifact_stats_share_no_key_with_the_memory_stats(store: pathlib.Path) -> None:
    """The disk layer reports under its own keys, never reusing ``cache_stats``'s."""
    A.reset_stats()
    disk = set(E.artifact_cache_stats())
    memory = set(E.cache_stats())
    dense = set(D.dense_cache_stats())
    assert disk.isdisjoint(memory), disk & memory
    assert disk.isdisjoint(dense), disk & dense
    assert len(memory) == 8, sorted(memory)
    assert all(key.startswith("artifact_") for key in disk), sorted(disk)
    assert {"artifact_enabled", "artifact_files", "artifact_hits",
            "artifact_misses", "artifact_load_failures", "artifact_stores",
            "artifact_save_failures"} == disk


def test_clear_artifacts_leaves_foreign_files_alone(store: pathlib.Path) -> None:
    """Only ``*.evoxcart`` (and its temp files) are deleted from the directory."""
    store.mkdir(parents=True)
    keep = store / "notes.txt"
    keep.write_text("not mine")
    temporary = store / ("graph" + A.SUFFIX + ".tmp-1-deadbeef")
    temporary.write_bytes(b"partial")
    real = store / ("graph" + A.SUFFIX)
    real.write_bytes(b"payload")

    A.clear()
    assert keep.read_text() == "not mine"
    assert not temporary.exists()
    assert not real.exists()
    assert A.stats()["artifact_enabled"] == 1


def test_clear_artifacts_is_a_noop_when_disabled(disabled: None) -> None:
    """With the switch off ``clear_artifacts()`` touches nothing and does not raise."""
    E.clear_artifacts()
    assert E.artifact_cache_stats()["artifact_enabled"] == 0
