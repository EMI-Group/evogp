#!/usr/bin/env python
"""``executable_memo_check.py`` -- dev CLI: the ``_executable`` program->Executable memo.

Small, self-contained check battery for the **in-process memo** in
:func:`evogp.tree.evoxc_engine._executable` (the per-tree "build-or-hit" step of the
specialized evaluation path).  It is **NOT** pytest-collected, is imported by
nothing, and only *reads* the library: it patches ``evoxc_engine.build_cached``
with a counting pass-through, calls the private helpers as they are, and swaps
``evoxc_engine._executable`` for a memo-free twin to get a differential reference.

Why it exists: ``evoxc.build_cached`` re-derives its cache key on every call --
even on a hit -- so per-tree lookups cost ~16 us/tree on the numpy backend while
a plain Python dict lookup on the very same (program, shape, backend, device,
options) tuple costs well under a microsecond.  The engine therefore keeps a
bounded value-keyed memo in front of ``CACHE``.  This battery pins the
properties that make that safe and worthwhile:

* **presence** -- repeated ``_executable`` calls with identical arguments must
  reach ``build_cached`` exactly ONCE, return the identical object (the very one
  ``CACHE`` holds), and a value-equal (but distinct) program object must
  memo-hit too (that is the elite/duplicate re-decode case the memo is for);
* **key discipline** -- the memo key must differ whenever any component that
  determines the executable differs (program, ``n_rows``, ``input_len``,
  backend name, device kind/index, options content), must be immutable/detached
  from the caller's option dict, and must not depend on container ordering --
  i.e. it is exactly as discriminating as ``build_cached``'s own key (mirrors
  ``evoxc.build_cache._canonicalize_options``);
* **hit cost** -- a memo hit must cost a fraction of a ``build_cached`` hit
  measured in the same process (the absolute number is printed, the *ratio* is
  the assertion, so a loaded box cannot turn this into a flake);
* plus the cache contract (``cache_stats`` counters, ``clear_cache`` clearing
  the memo, the LRU bound) and bit-exact equality of the memo path against a
  memo-free ``_executable`` for both ``SR_fitness`` and ``evaluate``.

Usage (repository root, project venv; a CUDA-visible device is needed only
because ``Forest.SR_fitness`` goes through ``check_tensor`` -> ``"cuda"``)::

    PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        src/evogp/tree/executable_memo_check.py --reps 200

Exit code 0 = every check passed, 1 = at least one failed, 2 = a hard error.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import evoxc  # noqa: E402
from evogp.tree import Forest  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402
from evogp.tree.warm_path_probe import _load_fixture  # noqa: E402  (sibling dev CLI, inert lib)

#: A memo hit is asserted to be at most this fraction of a ``build_cached`` hit.
_HIT_RATIO_MAX = 0.5
#: ... and, in absolute terms, below this many microseconds (>> the real ~0.5 us).
_HIT_US_MAX = 5.0


class Report:
    """Collects pass/fail entries and prints one line per check."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.measurements: Dict[str, float] = {}
        self.population: Any = None
        self.fixture: Dict[str, Any] = {}
        self.forest: Any = None

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.entries.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        return ok

    @property
    def ok(self) -> bool:
        return all(entry["ok"] for entry in self.entries)


class BuildCounter:
    """Pass-through wrapper around ``evoxc_engine.build_cached`` counting calls."""

    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.original(*args, **kwargs)


class CountingSection:
    """Context manager: count ``evoxc_engine.build_cached`` calls for its body."""

    def __init__(self, report: Report, name: str) -> None:
        self.report = report
        self.name = name
        self.counter = BuildCounter(E.build_cached)

    def __enter__(self) -> BuildCounter:
        E.build_cached = self.counter
        return self.counter

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        E.build_cached = self.counter.original
        if exc is not None:  # one bad check, not a crash
            self.report.add(self.name, False, f"{type(exc).__name__}: {exc}")
            return True
        return False


def _no_memo_executable(
    program: Any,
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Any:
    """The memo-free twin of ``E._executable`` (the pre-memo body, verbatim).

    Swapping it in for ``E._executable`` (``_run_tree`` resolves that name at
    call time) must not change a single bit of what the engine computes.
    """
    spec = evoxc.TensorSpec((int(n_rows), int(input_len)), evoxc.float32)
    if backend_name == "numpy":
        return _ORIGINAL_BUILD_CACHED(
            E._eval_step, spec, program, backend=backend_name, device=device, cache=E.CACHE
        )
    return _ORIGINAL_BUILD_CACHED(
        E._eval_step,
        spec,
        program,
        backend=backend_name,
        device=device,
        cache=E.CACHE,
        **options,
    )


#: The un-wrapped ``build_cached`` (the counter section patches the module global).
_ORIGINAL_BUILD_CACHED = E.build_cached


def _distinct_programs(pop: Any, want: int) -> List[Any]:
    """Up to ``want`` value-distinct static programs from a decoded population."""
    seen: Dict[Any, Any] = {}
    for tree in pop.trees:
        if tree.program is not None:
            seen.setdefault(tree.program, tree.program)
        if len(seen) >= want:
            break
    return list(seen.values())


def _equal_copy(program: Any) -> Any:
    """A value-equal but distinct program object (a fresh decode of one tree)."""
    return tuple(item if not isinstance(item, tuple) else tuple(item) for item in program)


def _median_us(fn: Callable[[], Any], reps: int) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e6


# ---------------------------------------------------------------------------
# check families
# ---------------------------------------------------------------------------


def check_presence(
    report: Report, program: Any, n_rows: int, input_len: int, backend: Any, reps: int
) -> None:
    """m*: repeated calls with identical arguments reach build_cached once."""
    print("-- presence (memo absorbs repeated lookups) --")
    with CountingSection(report, "m1-m4 memo presence") as counter:
        E.clear_cache()
        before = E.cache_stats()
        exe = E._executable(program, n_rows, input_len, *backend)
        again = [E._executable(program, n_rows, input_len, *backend) for _ in range(reps)]
        after = E.cache_stats()
        report.add(
            f"m1 one build_cached for {reps + 1} identical calls",
            counter.count == 1,
            f"build_cached calls={counter.count} (expected 1)",
        )
        report.add(
            "m2 every call returns the identical Executable",
            all(item is exe for item in again),
            f"{len(again)} calls, distinct objects={len({id(item) for item in again})}",
        )
        direct = _ORIGINAL_BUILD_CACHED(
            E._eval_step,
            evoxc.TensorSpec((int(n_rows), int(input_len)), evoxc.float32),
            program,
            backend=backend[0],
            device=backend[1],
            cache=E.CACHE,
        )
        report.add(
            "m3 the memo returns the very object CACHE holds",
            exe is direct,
            f"exe={type(exe).__name__}",
        )
        report.add(
            "m4 memo hits bypass CACHE (documented behavior)",
            after["hits"] - before["hits"] == 0,
            f"CACHE hits delta={after['hits'] - before['hits']}",
        )

    with CountingSection(report, "m5 value-keyed memo") as counter:
        E.clear_cache()
        E._executable(program, n_rows, input_len, *backend)
        counter.reset()
        hit = E._executable(_equal_copy(program), n_rows, input_len, *backend)
        same = E._executable(program, n_rows, input_len, *backend)
        report.add(
            "m5 a value-equal program object memo-hits (elite re-decode)",
            counter.count == 0 and hit is same,
            f"build_cached calls={counter.count} (expected 0), same object={hit is same}",
        )

    print("-- presence (distinct combinations) --")
    programs = _distinct_programs(report.population, 3)
    shapes = [(n_rows, input_len), (2 * n_rows, input_len)]
    with CountingSection(report, "m6/m7 distinct combinations") as counter:
        E.clear_cache()
        counter.reset()
        stable = True
        first: Dict[Tuple[int, int], Any] = {}
        for round_index in range(3):
            for program_index, item in enumerate(programs):
                for rows, cols in shapes:
                    got = E._executable(item, rows, cols, *backend)
                    key = (program_index, rows)
                    if round_index == 0:
                        first[key] = got
                    elif first[key] is not got:
                        stable = False
        expected = len(programs) * len(shapes)
        report.add(
            f"m6 {len(programs)} programs x {len(shapes)} shapes x 3 rounds "
            f"-> {expected} builds",
            counter.count == expected,
            f"build_cached calls={counter.count} (expected {expected})",
        )
        report.add(
            "m7 the memo is stable across rounds (per-combination identity)",
            stable,
            f"{len(first)} distinct combinations",
        )


def check_key(report: Report, program: Any, n_rows: int, input_len: int, backend: Any) -> None:
    """k*: the memo key is as discriminating as build_cached's key."""
    print("-- key discipline --")
    key_fn = getattr(E, "_executable_key", None)
    if key_fn is None:
        report.add(
            "k0 _executable_key is exposed",
            False,
            "evogp.tree.evoxc_engine has no _executable_key (memo not implemented)",
        )
        return
    device = backend[1]
    base = key_fn(program, n_rows, input_len, backend[0], device, dict(backend[2]))
    report.add(
        "k1 identical arguments key identically",
        base == key_fn(program, n_rows, input_len, backend[0], device, dict(backend[2])),
        "",
    )
    report.add(
        "k2 value-equal program objects share a key",
        base == key_fn(_equal_copy(program), n_rows, input_len, backend[0], device, dict(backend[2])),
        "",
    )
    other = _distinct_programs(report.population, 2)
    # The engine's resolved options carry ``opt_level`` ONLY -- ``target_backends``
    # is inferred from the resolved device by ``build_cached`` and is no longer a
    # knob the engine passes -- so the option cases below use that key (with a
    # level that differs from the resolved one) and a neutral list-valued key.
    resolved_level = dict(backend[2]).get("opt_level")
    alt_level = "O1" if resolved_level == "O0" else "O0"
    # ... and a device kind that differs from the resolved one: the GPU kind on
    # the numpy/cpu route, the CPU kind on an iree route that already resolved
    # the GPU kind (``Device("cuda", ...)`` does not exist on a ROCm host).
    other_kind = "cpu" if E._gpu_device_kind() == device.kind else E._gpu_device_kind()
    variants = {
        "another program": (
            other[1] if len(other) > 1 else program,
            n_rows,
            input_len,
            backend[0],
            device,
            dict(backend[2]),
        ),
        "n_rows+1": (program, n_rows + 1, input_len, backend[0], device, dict(backend[2])),
        "input_len+1": (program, n_rows, input_len + 1, backend[0], device, dict(backend[2])),
        "another backend": (program, n_rows, input_len, "iree-other", device, dict(backend[2])),
        "another device kind": (
            program, n_rows, input_len, backend[0], evoxc.core.Device(other_kind, 0),
            dict(backend[2]),
        ),
        "another device index": (
            program,
            n_rows,
            input_len,
            backend[0],
            evoxc.core.Device(device.kind, device.index + 1),
            dict(backend[2]),
        ),
        "another option value": (
            program,
            n_rows,
            input_len,
            backend[0],
            device,
            {"opt_level": alt_level},
        ),
        "an extra option": (
            program, n_rows, input_len, backend[0], device, {**dict(backend[2]), "extra": 1},
        ),
    }
    for label, args in variants.items():
        report.add(f"k3 {label} changes the key", key_fn(*args) != base, "")
    report.add(
        "k4 option dict insertion order does not change the key",
        key_fn(
            program, n_rows, input_len, backend[0], device,
            {"opt_level": alt_level, "split_reduction": True},
        )
        == key_fn(
            program, n_rows, input_len, backend[0], device,
            {"split_reduction": True, "opt_level": alt_level},
        ),
        "",
    )
    report.add(
        "k5 option list order does not change the key",
        key_fn(program, n_rows, input_len, backend[0], device, {"tags": ["a", "b"]})
        == key_fn(program, n_rows, input_len, backend[0], device, {"tags": ["b", "a"]}),
        "",
    )
    report.add(
        "k6 option list/tuple container type does not change the key",
        key_fn(program, n_rows, input_len, backend[0], device, {"tags": ["a", "b"]})
        == key_fn(program, n_rows, input_len, backend[0], device, {"tags": ("a", "b")}),
        "",
    )
    options = {"opt_level": alt_level, "tags": ["a"]}
    key_before = key_fn(program, n_rows, input_len, backend[0], device, options)
    options["opt_level"] = "O2"  # a different (and different-from-alt) level
    options["tags"].append("b")
    report.add(
        "k7 the key is immutable / detached from the caller's options dict",
        key_before
        == key_fn(
            program, n_rows, input_len, backend[0], device,
            {"opt_level": alt_level, "tags": ["a"]},
        ),
        "",
    )


def check_hit_cost(
    report: Report, program: Any, n_rows: int, input_len: int, backend: Any, reps: int
) -> None:
    """c*: a memo hit must be far cheaper than a build_cached hit."""
    print("-- hit cost --")
    spec = evoxc.TensorSpec((int(n_rows), int(input_len)), evoxc.float32)
    E.clear_cache()
    E._executable(program, n_rows, input_len, *backend)  # populate the memo
    _ORIGINAL_BUILD_CACHED(
        E._eval_step, spec, program, backend=backend[0], device=backend[1], cache=E.CACHE
    )

    memo_us = _median_us(lambda: E._executable(program, n_rows, input_len, *backend), reps)
    build_us = _median_us(
        lambda: _ORIGINAL_BUILD_CACHED(
            E._eval_step, spec, program, backend=backend[0], device=backend[1], cache=E.CACHE
        ),
        reps,
    )
    ratio = memo_us / build_us if build_us > 0 else float("inf")
    report.add(
        f"c1 memo hit <= {_HIT_RATIO_MAX:g}x a build_cached hit",
        ratio <= _HIT_RATIO_MAX,
        f"memo={memo_us:.3f} us build_cached={build_us:.3f} us ratio={ratio:.3f}",
    )
    report.add(
        f"c2 memo hit < {_HIT_US_MAX:g} us absolute",
        memo_us < _HIT_US_MAX,
        f"memo={memo_us:.3f} us ({reps} reps, median)",
    )
    report.measurements.update(
        {"memo_hit_us": memo_us, "build_cached_hit_us": build_us, "ratio": ratio}
    )


def check_cache_contract(
    report: Report, program: Any, n_rows: int, input_len: int, backend: Any, reps: int
) -> None:
    """s*/b*: stats, clear_cache and the LRU bound."""
    print("-- cache contract --")
    stats = E.cache_stats()
    memo_keys = ("memo_hits", "memo_misses", "memo_entries", "memo_maxsize")
    if not report.add(
        "s1 cache_stats exposes the memo counters",
        all(name in stats for name in memo_keys),
        f"keys={sorted(stats)}",
    ):
        return
    memo = getattr(E, "_EXECUTABLE_MEMO", None)
    report.add(
        "b1 the memo is bounded by the compile cache's maxsize",
        memo is not None and int(stats["memo_maxsize"]) == int(E.CACHE.maxsize),
        f"memo_maxsize={stats['memo_maxsize']} CACHE.maxsize={E.CACHE.maxsize}",
    )
    with CountingSection(report, "s2-s4 memo counters and clear_cache") as counter:
        E.clear_cache()
        base = E.cache_stats()
        report.add(
            "s2 clear_cache() drops the memo entries and resets its counters",
            base["memo_entries"] == 0
            and base["memo_hits"] == 0
            and base["memo_misses"] == 0,
            f"after clear_cache: entries={base['memo_entries']} hits={base['memo_hits']} "
            f"misses={base['memo_misses']}",
        )
        E._executable(program, n_rows, input_len, *backend)
        for _ in range(reps):
            E._executable(program, n_rows, input_len, *backend)
        got = E.cache_stats()
        report.add(
            "s3 counters account for every lookup",
            got["memo_misses"] - base["memo_misses"] == 1
            and got["memo_hits"] - base["memo_hits"] == reps
            and got["memo_entries"] - base["memo_entries"] == 1,
            f"memo misses=+{got['memo_misses'] - base['memo_misses']} "
            f"hits=+{got['memo_hits'] - base['memo_hits']} "
            f"entries=+{got['memo_entries'] - base['memo_entries']}",
        )
        E.clear_cache()
        counter.reset()
        E._executable(program, n_rows, input_len, *backend)
        report.add(
            "s4 clear_cache() really invalidates the memo",
            counter.count == 1 and E.cache_stats()["memo_misses"] == 1,
            f"build_cached calls after clear={counter.count}",
        )
    if memo is None:
        return
    programs = _distinct_programs(report.population, 5)
    if len(programs) < 5:
        report.add("b2 LRU bound", False, f"only {len(programs)} distinct programs in the fixture")
        return
    with CountingSection(report, "b2 LRU bound") as counter:
        old_maxsize = memo.maxsize
        try:
            memo.maxsize = 3
            E.clear_cache()
            keys = []
            for item in programs[:5]:
                E._executable(item, n_rows, input_len, *backend)
                keys.append(
                    E._executable_key(
                        item, n_rows, input_len, backend[0], backend[1], backend[2]
                    )
                )
            held = len(memo)
            evicted = [key for key in keys[:2] if key not in memo]
            counter.reset()
            E._executable(programs[0], n_rows, input_len, *backend)  # the LRU victim
            report.add(
                "b2 the memo never exceeds maxsize and evicts LRU-first",
                held == 3 and len(evicted) == 2 and counter.count == 1,
                f"entries={held} evicted={len(evicted)} rebuilds={counter.count}",
            )
        finally:
            memo.maxsize = old_maxsize
            E.clear_cache()


def check_differential(report: Report, reps: int) -> None:
    """d*: the memo path must be bit-identical to the memo-free engine."""
    print("-- differential (memo vs no memo) --")
    fixture, forest = report.fixture, report.forest
    x, labels = fixture["x"].to("cuda"), fixture["labels"].to("cuda")
    pop = report.population
    forest.SR_fitness(x, labels, True, "auto")  # warm

    memo_fitness = forest.SR_fitness(x, labels, True, "auto").detach().cpu().numpy()
    memo_eval = E.evaluate(pop, x, per_row=False).detach().cpu().numpy()
    E.clear_cache()  # a full rebuild must reproduce the same bits
    cold_fitness = forest.SR_fitness(x, labels, True, "auto").detach().cpu().numpy()

    E._executable = _no_memo_executable  # type: ignore[assignment]
    try:
        plain_eval = E.evaluate(pop, x, per_row=False).detach().cpu().numpy()
        E.clear_cache()
        plain_fitness = forest.SR_fitness(x, labels, True, "auto").detach().cpu().numpy()
    finally:
        E._executable = _executable_ref  # type: ignore[assignment]

    def non_bit(a: np.ndarray, b: np.ndarray) -> int:
        left, right = a.view(np.int32), b.view(np.int32)
        return int((left != right).sum())

    report.add(
        "d1 memo and memo-free SR_fitness are bit-identical",
        np.array_equal(memo_fitness.view(np.int32), plain_fitness.view(np.int32)),
        f"non-bit-exact={non_bit(memo_fitness, plain_fitness)}/{memo_fitness.size}",
    )
    report.add(
        "d2 memo and memo-free evaluate are bit-identical",
        np.array_equal(memo_eval.view(np.int32), plain_eval.view(np.int32)),
        f"non-bit-exact={non_bit(memo_eval, plain_eval)}/{memo_eval.size}",
    )
    report.add(
        "d3 a cleared memo reproduces the same fitness bits",
        np.array_equal(cold_fitness.view(np.int32), memo_fitness.view(np.int32)),
        "",
    )


def check_warm_end_to_end(report: Report, reps: int) -> Dict[str, float]:
    """One warm ``Forest.SR_fitness`` headline for the record (report-only)."""
    fixture, forest = report.fixture, report.forest
    x, labels = fixture["x"].to("cuda"), fixture["labels"].to("cuda")
    forest.SR_fitness(x, labels, True, "auto")
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        forest.SR_fitness(x, labels, True, "auto")
        times.append(time.perf_counter() - start)
    median_ms = statistics.median(times) * 1e3
    print(
        f"[info] warm Forest.SR_fitness: {median_ms:.3f} ms/call "
        f"(pop={forest.pop_size}, rows={int(x.shape[0])}, {reps} reps)"
    )
    return {"warm_sr_fitness_ms": median_ms}


#: Saved in ``main`` so the differential section can restore ``E._executable``.
_executable_ref: Callable[..., Any] = E._executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="test/goldens/forest_random.npz")
    parser.add_argument("--reps", type=int, default=200, help="timing loop repetitions")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    fixture = _load_fixture(args.fixture)
    input_len = fixture["input_len"]
    output_len = int(fixture["labels"].shape[1])
    forest = Forest(
        input_len,
        output_len,
        fixture["node_value"].to("cpu"),
        fixture["node_type"].to("cpu"),
        fixture["subtree_size"].to("cpu"),
    )
    pop = E.decode_population(
        fixture["node_value"], fixture["node_type"], fixture["subtree_size"],
        input_len, output_len,
    )
    programs = _distinct_programs(pop, 5)
    if not programs:
        raise SystemExit(f"{args.fixture}: no compilable tree in the fixture")
    program = programs[0]
    backend = E._resolve_backend(None, None)
    n_rows = int(fixture["x"].shape[0])
    rounds = max(4, min(args.reps, 25))

    global _executable_ref
    _executable_ref = E._executable

    report = Report()
    report.population = pop
    report.fixture = fixture
    report.forest = forest

    print(
        f"{args.fixture} [{backend[0]} / {backend[1]}] pop={forest.pop_size} "
        f"tree={forest.max_tree_len} rows={n_rows} distinct programs>={len(programs)}"
    )
    check_presence(report, program, n_rows, input_len, backend, rounds)
    check_key(report, program, n_rows, input_len, backend)
    check_hit_cost(report, program, n_rows, input_len, backend, args.reps)
    check_cache_contract(report, program, n_rows, input_len, backend, rounds)
    check_differential(report, args.reps)
    report.measurements.update(check_warm_end_to_end(report, rounds))

    failed = [entry["check"] for entry in report.entries if not entry["ok"]]
    print(
        f"\n{len(report.entries) - len(failed)}/{len(report.entries)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "fixture": args.fixture,
                    "backend": backend[0],
                    "pop_size": forest.pop_size,
                    "rows": n_rows,
                    "measurements": report.measurements,
                    "checks": report.entries,
                },
                handle,
                indent=2,
            )
    return 0 if report.ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # hard error: exit 2, as documented
        import traceback

        traceback.print_exc()
        print(f"hard error: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
