#!/usr/bin/env python
"""``stage_probe.py`` -- per-stage time/event decomposition of ``problem.evaluate``.

R2 add-on for ``bench_gp.py`` (``--stage-probe``), **new side only**.  The legacy
engine is one fused CUDA call per population; the evoxc "true jit" engine is a
per-tree host decode + ``evoxc.build_cached`` + ``evoxc.run`` + host readback, so a
whole-call wall clock cannot say *where* the time goes.  This module wraps the
engine's hot call sites at **runtime** (``src/`` is never edited, and nothing is
imported from the library either) and books every evaluate call / training
generation into named stages.

The wrapped chain (confirmed against ``src/evogp/tree/evoxc_engine.py`` /
``forest.py``)::

    problem.evaluate(forest)                    <- harness timing (outer total)
      Forest.SR_fitness            forest.py
        evoxc_engine.decode_population           -> decode_ms
        evoxc_engine.sr_fitness
          evoxc_engine.evaluate
            evoxc_engine._run_tree               -> per-tree residual = readback_ms
              evoxc_engine._executable
                evoxc.build_cached                -> build_{hit,miss}_ms + counts
              evoxc.run                           -> run_ms / run_calls
          (loss_ms  = sr_fitness_ms - evaluate_ms)
    (misc_ms = total_ms - decode_ms - sr_fitness_ms)

Stage semantics (one record per evaluate call / training generation)::

    total_ms     harness-measured outer wall clock (bench_gp.py per_call_ms /
                 per_gen_ms; the probe's own bracket is kept as probe_span_ms)
    probe_span_ms  begin() -> snapshot() inside the probe (>= total_ms; the extra
                 is the harness' own torch.cuda.synchronize() pair)
    decode_ms    time in ``decode_population`` (host-side population decode;
                 evoxc tracing of the per-tree static programs)
    build_hit_ms / build_miss_ms
                 time in ``build_cached``, split by the ``evoxc.CompileCache``
                 counter delta around the call (a hit is a dict lookup, a miss is
                 trace+lower+compile+load): ``build_hits`` / ``build_misses``
    run_ms       time in ``evoxc.run`` (``run_calls`` = number of invocations)
    readback_ms  per-tree residual inside ``_run_tree`` =
                 tree_total - build - run, i.e. the device->host result readback
                 (+ its synchronization) and the host glue around it;
                 ``readback_calls`` = trees actually invoked (constant/collapsed
                 trees are skipped by the engine, hence ``trees_invoked`` <=
                 ``trees_total``)
    upload_ms    residual inside ``evaluate`` = evaluate_ms - build - run -
                 readback: the final ``torch.from_numpy(rows)...to(device)``
                 result upload plus torch glue
    loss_ms      sr_fitness_ms - evaluate_ms (label upload + the error reduction)
    misc_ms      total_ms - decode_ms - sr_fitness_ms (harness/torch glue, and in
                 training mode the crossover/mutation/selection operators)

Device traffic counters (measured, not derived):

* ``h2d_calls`` -- ``evoxc.from_numpy`` invocations made *inside* ``_run_tree`` and
  outside a nested ``evoxc.run`` (the engine's one explicit input upload per
  evaluate call on a device backend; ``0`` for the numpy backend);
* ``d2h_calls`` -- ``_run_tree`` invocations whose backend is not ``numpy``, i.e.
  the per-tree ``result.to(cpu).numpy()`` readbacks.  Torch-level host<->device
  copies outside the engine (e.g. ``sr_fitness``' label tensor) are *not* counted;
  they land in ``loss_ms``/``upload_ms``.

Per-rep totals (``<stages>.reps[i].totals``) separate cold from warm work:
``cold_builds``/``cold_build_ms`` (the misses of the first record = the compile
tax), ``total_build_misses``/``total_build_miss_ms`` (whole rep) and
``warm_misses``/``warm_build_miss_ms`` (misses *after* the first record -- a
warm pass is proven compile-free when these are 0), plus the summed stage ms and
event counts.

Robustness: every wrapper is a pure pass-through (identical return value and
identical exception behaviour, argument inspection only), the accumulator is a
plain single-threaded dict of raw ``time.perf_counter()`` floats with no per-tree
allocation, and a missing attribute degrades instead of crashing -- the stage
keeps reporting 0, ``<stages>.partial`` becomes ``true`` and ``note`` names what
was not wrapped.

API (kept small on purpose; ``bench_gp.py`` needs ~5 hooks per call site)::

    install(engine_module, evoxc_module=None) -> status dict (idempotent)
    container(side=None)          -> the ``stages`` JSON block (holds ``reps``)
    status_line()                 -> one-line install report
    begin_rep(rep, kind)          -> per-rep block (resets ``begin``/``snapshot``)
    begin()                       -> start a new record (call it before the timed
                                     region of one evaluate call / generation)
    snapshot(i=None, outer_ms=None) -> finalized record (appends to the rep block)
    finish_rep()                  -> writes ``totals``, returns the summary line
    totals(records=None)          -> the per-rep totals dict
    records() / reps() / status() -> read-only access for report tooling
    reset()                       -> drop all state (keeps the install)

Self-test (no torch, no GPU, no evoxc: a fake engine with the same call shape)::

    python test/benchmark/stage_probe.py
"""

from __future__ import annotations

import inspect
import time

PROBE_NAME = "stage_probe/1"

_NOW = time.perf_counter
_MS = 1000.0

#: accumulator keys (raw seconds / ints; one dict per evaluate call)
_ACC = (
    "decode_s",
    "decode_calls",
    "sr_fitness_s",
    "sr_fitness_calls",
    "evaluate_s",
    "evaluate_calls",
    "build_s",
    "build_calls",
    "build_hit_s",
    "build_hits",
    "build_miss_s",
    "build_misses",
    "run_s",
    "run_calls",
    "readback_s",
    "readback_calls",
    "tree_s",
    "trees_invoked",
    "trees_total",
    "h2d_calls",
    "d2h_calls",
)

_INT_ACC = (
    "decode_calls",
    "sr_fitness_calls",
    "evaluate_calls",
    "build_calls",
    "build_hits",
    "build_misses",
    "run_calls",
    "readback_calls",
    "trees_invoked",
    "trees_total",
    "h2d_calls",
    "d2h_calls",
)

_SUM_MS_KEYS = (
    "total_ms",
    "probe_span_ms",
    "decode_ms",
    "sr_fitness_ms",
    "evaluate_ms",
    "loss_ms",
    "misc_ms",
    "build_ms",
    "build_hit_ms",
    "build_miss_ms",
    "run_ms",
    "readback_ms",
    "upload_ms",
)

_SUM_INT_KEYS = _INT_ACC

# ---------------------------------------------------------------------------
# state (single-threaded by contract: bench_gp.py is one thread)
# ---------------------------------------------------------------------------

_STATUS = None  # install() result; also the live counters container
_COUNTERS = None  # callable -> (hits, misses), or None
_RUN_TREE_BACKEND_POS = None  # positional index of ``backend_name`` in _run_tree

_CONTAINER = None  # the ``stages`` JSON block
_REP = None  # current per-rep block
_RECORDS = []  # = _REP["records"] once begin_rep ran
_CUR = None  # current record accumulator
_SPAN_T0 = None  # begin() timestamp

_IN_TREE = False  # inside _run_tree
_IN_BUILD = False  # inside build_cached
_IN_RUN = False  # inside evoxc.run
_TREE_BUILD_S = 0.0  # build time inside the current _run_tree (for the residual)
_TREE_RUN_S = 0.0  # evoxc.run time inside the current _run_tree


def _new_acc():
    return {key: (0 if key in _INT_ACC else 0.0) for key in _ACC}


# ---------------------------------------------------------------------------
# install: wrap the engine's call sites
# ---------------------------------------------------------------------------


def _make_counters(engine_module):
    """``(read, label)`` where ``read() -> (hits, misses)``, or ``(None, None)``."""
    cache = getattr(engine_module, "CACHE", None)
    if cache is not None and hasattr(cache, "hits") and hasattr(cache, "misses"):
        def read(cache=cache):
            return int(cache.hits), int(cache.misses)

        return read, "CACHE.hits/CACHE.misses"
    stats = getattr(engine_module, "cache_stats", None)
    if callable(stats):
        def read(stats=stats):
            info = stats()
            return int(info["hits"]), int(info["misses"])

        return read, "cache_stats()"
    return None, None


def _patch(target, name, factory, status, required=True, note=None):
    """Replace ``target.name`` by ``factory(original)``; never raises."""
    if target is None:
        status["missing"].append(name)
        if required:
            status["partial"] = True
        return False
    try:
        original = getattr(target, name, None)
    except Exception:  # a module __getattr__ hook may raise; treat as missing
        original = None
    if original is None or not callable(original):
        status["missing"].append(name)
        if required:
            status["partial"] = True
        return False
    if getattr(original, "_stage_probe_wrapped", False):
        return True  # already ours (idempotent install)
    wrapper = factory(original)
    try:
        wrapper._stage_probe_wrapped = True
        wrapper.__wrapped__ = original
        wrapper.__name__ = getattr(original, "__name__", name)
        wrapper.__doc__ = getattr(original, "__doc__", None)
    except (AttributeError, TypeError):
        pass
    try:
        setattr(target, name, wrapper)
    except Exception:
        status["missing"].append(name)
        if required:
            status["partial"] = True
        return False
    status["wrapped"].append(name if note is None else note)
    return True


def install(engine_module, evoxc_module=None):
    """Wrap ``engine_module``'s evaluate chain; idempotent, never raises.

    Args:
        engine_module: the ``evogp.tree.evoxc_engine`` module (its module globals
            are what ``forest.py`` and the engine itself call, so patching them
            covers every call site).
        evoxc_module: the ``evoxc`` package (default: ``engine_module.evoxc``, then
            ``import evoxc``); only ``evoxc.run`` / ``evoxc.from_numpy`` are wrapped.

    Returns:
        The status dict (also stored on ``status()``): ``installed``,
        ``partial``, ``note``, ``wrapped``, ``missing``, ``counters``,
        ``nested_runs`` (``evoxc.run`` calls seen *inside* a build).
    """
    global _STATUS, _COUNTERS, _RUN_TREE_BACKEND_POS
    if _STATUS is not None:
        return _STATUS
    status = {
        "probe": PROBE_NAME,
        "installed": False,
        "partial": False,
        "note": "",
        "engine_module": getattr(engine_module, "__name__", repr(engine_module)),
        "evoxc_module": None,
        "wrapped": [],
        "missing": [],
        "counters": None,
        "nested_runs": 0,
    }
    _STATUS = status  # wrappers read the nested-run counter through this

    if evoxc_module is None:
        evoxc_module = getattr(engine_module, "evoxc", None)
    if evoxc_module is None:
        try:
            import evoxc as evoxc_module  # type: ignore[no-redef]
        except Exception as exc:
            evoxc_module = None
            status["note"] = f"evoxc not importable ({type(exc).__name__}: {exc})"
    status["evoxc_module"] = getattr(evoxc_module, "__name__", None)

    _patch(engine_module, "decode_population", _wrap_decode, status)
    _patch(engine_module, "_run_tree", _wrap_run_tree, status)
    _patch(engine_module, "build_cached", _wrap_build_cached, status)
    _patch(evoxc_module, "run", _wrap_evoxc_run, status)
    _patch(evoxc_module, "from_numpy", _wrap_from_numpy, status)
    _patch(engine_module, "evaluate", _wrap_evaluate, status)
    _patch(engine_module, "sr_fitness", _wrap_sr_fitness, status)

    read, label = _make_counters(engine_module)
    if read is None:
        status["missing"].append("cache_counters")
        status["partial"] = True
    else:
        _COUNTERS = read
        status["counters"] = label

    original = getattr(getattr(engine_module, "_run_tree", None), "__wrapped__", None)
    if original is not None:
        try:
            params = list(inspect.signature(original).parameters)
            _RUN_TREE_BACKEND_POS = params.index("backend_name") if "backend_name" in params else None
        except (TypeError, ValueError):
            _RUN_TREE_BACKEND_POS = None
    if _RUN_TREE_BACKEND_POS is None:
        status["missing"].append("_run_tree.backend_name")
        status["partial"] = True

    status["installed"] = "build_cached" in status["wrapped"] and "run" in status["wrapped"]
    if status["missing"]:
        status["partial"] = True
        missing = ", ".join(status["missing"])
        status["note"] = (
            f"{status['note']}; " if status["note"] else ""
        ) + f"not wrapped: {missing} (those stages stay 0, this block is partial)"
    return status


# ---------------------------------------------------------------------------
# wrappers (pure pass-through)
# ---------------------------------------------------------------------------


def _wrap_decode(original):
    def wrapper(*args, **kwargs):
        acc = _CUR
        t0 = _NOW()
        try:
            population = original(*args, **kwargs)
        except BaseException:
            if acc is not None:
                acc["decode_s"] += _NOW() - t0
            raise
        if acc is not None:
            acc["decode_s"] += _NOW() - t0
            acc["decode_calls"] += 1
            size = getattr(population, "pop_size", None)
            if isinstance(size, int):
                acc["trees_total"] += size
        return population

    return wrapper


def _wrap_run_tree(original):
    def wrapper(*args, **kwargs):
        global _IN_TREE, _TREE_BUILD_S, _TREE_RUN_S
        acc = _CUR
        t0 = _NOW()
        _TREE_BUILD_S = 0.0
        _TREE_RUN_S = 0.0
        _IN_TREE = True
        try:
            result = original(*args, **kwargs)
        except BaseException:
            _IN_TREE = False
            raise
        elapsed = _NOW() - t0
        _IN_TREE = False
        if acc is not None:
            acc["tree_s"] += elapsed
            acc["readback_s"] += elapsed - _TREE_BUILD_S - _TREE_RUN_S
            acc["readback_calls"] += 1
            acc["trees_invoked"] += 1
            if _backend_of(args, kwargs) not in (None, "numpy"):
                acc["d2h_calls"] += 1
        return result

    return wrapper


def _wrap_build_cached(original):
    def wrapper(*args, **kwargs):
        global _IN_BUILD, _TREE_BUILD_S
        acc = _CUR
        before = _COUNTERS() if (acc is not None and _COUNTERS is not None) else None
        t0 = _NOW()
        _IN_BUILD = True
        try:
            executable = original(*args, **kwargs)
        except BaseException:
            _IN_BUILD = False
            raise
        _IN_BUILD = False
        elapsed = _NOW() - t0
        if acc is not None:
            acc["build_s"] += elapsed
            acc["build_calls"] += 1
            if _IN_TREE:
                _TREE_BUILD_S += elapsed
            misses = None
            if before is not None:
                after = _COUNTERS()
                misses = after[1] - before[1]
            if misses is None or misses <= 0:  # unknown counters or a cache hit
                acc["build_hit_s"] += elapsed
                acc["build_hits"] += 1
            else:
                acc["build_miss_s"] += elapsed
                acc["build_misses"] += int(misses)
        return executable

    return wrapper


def _wrap_evoxc_run(original):
    def wrapper(*args, **kwargs):
        global _IN_RUN, _TREE_RUN_S
        if _IN_BUILD:  # a compile-internal run belongs to the build stage
            if _STATUS is not None:
                _STATUS["nested_runs"] += 1
            return original(*args, **kwargs)
        acc = _CUR
        t0 = _NOW()
        _IN_RUN = True
        try:
            result = original(*args, **kwargs)
        except BaseException:
            _IN_RUN = False
            if acc is not None:
                acc["run_s"] += _NOW() - t0
            raise
        _IN_RUN = False
        elapsed = _NOW() - t0
        if _IN_TREE:
            _TREE_RUN_S += elapsed
        if acc is not None:
            acc["run_s"] += elapsed
            acc["run_calls"] += 1
        return result

    return wrapper


def _wrap_from_numpy(original):
    def wrapper(*args, **kwargs):
        # The engine's explicit host->device upload: only inside _run_tree and
        # outside a nested evoxc.run (evoxc's own conversions are not counted).
        if _IN_TREE and not _IN_RUN and not _IN_BUILD:
            acc = _CUR
            if acc is not None:
                acc["h2d_calls"] += 1
        return original(*args, **kwargs)

    return wrapper


def _wrap_evaluate(original):
    def wrapper(*args, **kwargs):
        acc = _CUR
        t0 = _NOW()
        try:
            result = original(*args, **kwargs)
        except BaseException:
            if acc is not None:
                acc["evaluate_s"] += _NOW() - t0
            raise
        if acc is not None:
            acc["evaluate_s"] += _NOW() - t0
            acc["evaluate_calls"] += 1
            if acc["trees_total"] == 0 and args:
                size = getattr(args[0], "pop_size", None)
                if isinstance(size, int):
                    acc["trees_total"] += size
        return result

    return wrapper


def _wrap_sr_fitness(original):
    def wrapper(*args, **kwargs):
        acc = _CUR
        t0 = _NOW()
        try:
            result = original(*args, **kwargs)
        except BaseException:
            if acc is not None:
                acc["sr_fitness_s"] += _NOW() - t0
            raise
        if acc is not None:
            acc["sr_fitness_s"] += _NOW() - t0
            acc["sr_fitness_calls"] += 1
        return result

    return wrapper


def _backend_of(args, kwargs):
    """``backend_name`` of a ``_run_tree`` call (positional or keyword)."""
    value = kwargs.get("backend_name")
    if value is None and _RUN_TREE_BACKEND_POS is not None and len(args) > _RUN_TREE_BACKEND_POS:
        value = args[_RUN_TREE_BACKEND_POS]
    if value is None:
        for item in args:  # signature changed: first string argument wins
            if isinstance(item, str):
                return item
    return value


# ---------------------------------------------------------------------------
# records / reps / totals
# ---------------------------------------------------------------------------


def reset():
    """Drop every record and counter (the wrappers stay installed)."""
    global _CUR, _SPAN_T0, _REP, _RECORDS, _CONTAINER
    global _IN_TREE, _IN_BUILD, _IN_RUN, _TREE_BUILD_S, _TREE_RUN_S
    _CUR = None
    _SPAN_T0 = None
    _REP = None
    _RECORDS = []
    _CONTAINER = None
    _IN_TREE = _IN_BUILD = _IN_RUN = False
    _TREE_BUILD_S = _TREE_RUN_S = 0.0


def container(side=None):
    """Create (or return) the ``stages`` JSON block; ``side`` is recorded."""
    global _CONTAINER
    _CONTAINER = {
        "probe": PROBE_NAME,
        "side": side,
        "partial": bool(_STATUS.get("partial", True)) if _STATUS else True,
        "note": _STATUS.get("note", "") if _STATUS else "install() was never called",
        "status": _STATUS,
        "reps": [],
    }
    return _CONTAINER


def status():
    return _STATUS


def status_line():
    """One-line install report (for the harness log)."""
    if _STATUS is None:
        return f"stage probe {PROBE_NAME}: NOT installed (install() never called)"
    wrapped = ",".join(_STATUS["wrapped"]) or "<none>"
    line = (
        f"stage probe {PROBE_NAME}: installed={_STATUS['installed']} "
        f"partial={_STATUS['partial']} wrapped=[{wrapped}] "
        f"counters={_STATUS['counters']}"
    )
    if _STATUS["missing"]:
        line += f" missing=[{','.join(_STATUS['missing'])}]"
    if _STATUS["note"]:
        line += f" note={_STATUS['note']}"
    return line


def begin_rep(rep, kind):
    """Start a per-rep block (appended to the ``stages`` container)."""
    global _REP, _RECORDS, _CUR, _SPAN_T0
    block = {"rep": int(rep), "kind": str(kind), "records": [], "totals": None}
    if _CONTAINER is None:
        container()
    _CONTAINER["reps"].append(block)
    _REP = block
    _RECORDS = block["records"]
    _CUR = None
    _SPAN_T0 = None
    return block


def begin():
    """Start a new record; call it immediately before one timed call/generation."""
    global _CUR, _SPAN_T0
    _CUR = _new_acc()
    _SPAN_T0 = _NOW()
    return _CUR


def snapshot(i=None, outer_ms=None, label=None):
    """Finalize and append the current record; returns the record dict."""
    global _CUR, _SPAN_T0
    acc = _CUR if _CUR is not None else _new_acc()
    span_ms = None if _SPAN_T0 is None else (_NOW() - _SPAN_T0) * _MS
    record = _finalize(acc, i, outer_ms, span_ms, label)
    _RECORDS.append(record)
    _CUR = None
    _SPAN_T0 = None
    return record


def _finalize(acc, i, outer_ms, span_ms, label):
    decode_ms = _ms(acc["decode_s"])
    sr_ms = _ms(acc["sr_fitness_s"])
    evaluate_ms = _ms(acc["evaluate_s"])
    build_ms = _ms(acc["build_s"])
    run_ms = _ms(acc["run_s"])
    readback_ms = _ms(max(0.0, acc["readback_s"]))
    if outer_ms is not None:
        total_ms = round(float(outer_ms), 3)
    elif span_ms is not None:
        total_ms = round(span_ms, 3)
    else:
        total_ms = round(decode_ms + sr_ms, 3)
    record = {
        "i": None if i is None else int(i),
        "total_ms": total_ms,
        "probe_span_ms": None if span_ms is None else round(span_ms, 3),
        "decode_ms": decode_ms,
        "decode_calls": int(acc["decode_calls"]),
        "sr_fitness_ms": sr_ms,
        "sr_fitness_calls": int(acc["sr_fitness_calls"]),
        "evaluate_ms": evaluate_ms,
        "evaluate_calls": int(acc["evaluate_calls"]),
        "loss_ms": round(max(0.0, sr_ms - evaluate_ms), 3),
        "misc_ms": round(max(0.0, total_ms - decode_ms - sr_ms), 3),
        "build_ms": build_ms,
        "build_calls": int(acc["build_calls"]),
        "build_hit_ms": _ms(acc["build_hit_s"]),
        "build_hits": int(acc["build_hits"]),
        "build_miss_ms": _ms(acc["build_miss_s"]),
        "build_misses": int(acc["build_misses"]),
        "run_ms": run_ms,
        "run_calls": int(acc["run_calls"]),
        "readback_ms": readback_ms,
        "readback_calls": int(acc["readback_calls"]),
        "upload_ms": round(max(0.0, evaluate_ms - build_ms - run_ms - readback_ms), 3),
        "trees_total": int(acc["trees_total"]),
        "trees_invoked": int(acc["trees_invoked"]),
        "h2d_calls": int(acc["h2d_calls"]),
        "d2h_calls": int(acc["d2h_calls"]),
    }
    if label is not None:
        record["label"] = str(label)
    return record


def _ms(seconds):
    """Seconds -> milliseconds (rounded to 1e-3 ms; ``None`` stays ``None``)."""
    return None if seconds is None else round(seconds * _MS, 3)


def records():
    """The records of the current (or last) rep."""
    return _RECORDS


def reps():
    """The per-rep blocks collected so far."""
    return [] if _CONTAINER is None else _CONTAINER["reps"]


def totals(records=None):
    """Per-rep totals: cold/warm build separation + summed stages and counters."""
    recs = list(_RECORDS if records is None else records)
    first = recs[0] if recs else None
    out = {
        "calls": len(recs),
        "cold_builds": None if first is None else first["build_misses"],
        "cold_build_ms": None if first is None else first["build_miss_ms"],
        "total_build_misses": sum(r["build_misses"] for r in recs),
        "total_build_miss_ms": round(sum(r["build_miss_ms"] for r in recs), 3),
        "warm_misses": sum(r["build_misses"] for r in recs[1:]),
        "warm_build_miss_ms": round(sum(r["build_miss_ms"] for r in recs[1:]), 3),
    }
    for key in _SUM_MS_KEYS:
        out[key] = round(sum(r[key] for r in recs), 3)
    for key in _SUM_INT_KEYS:
        out[key] = sum(r[key] for r in recs)
    return out


def summary_line(rep):
    """Compact one-line digest of a finished rep block (``finish_rep`` return)."""
    t = rep.get("totals")
    if not t:
        return None
    def fmt(value):
        return "n/a" if value is None else f"{value:.1f}"

    return (
        f"stages[rep{rep['rep']}/{rep['kind']}] {t['calls']} calls: "
        f"cold {t['cold_builds']} miss {fmt(t['cold_build_ms'])}ms / warm {t['warm_misses']} miss "
        f"{fmt(t['warm_build_miss_ms'])}ms | decode {fmt(t['decode_ms'])} + "
        f"build-hit {fmt(t['build_hit_ms'])} ({t['build_hits']}) + "
        f"build-miss {fmt(t['build_miss_ms'])} ({t['build_misses']}) + "
        f"run {fmt(t['run_ms'])} ({t['run_calls']}) + "
        f"readback {fmt(t['readback_ms'])} ({t['readback_calls']}) + "
        f"upload {fmt(t['upload_ms'])} + loss {fmt(t['loss_ms'])} + misc {fmt(t['misc_ms'])} ms"
    )


def finish_rep():
    """Store ``totals`` on the current rep block; returns the summary line."""
    if _REP is None:
        return None
    _REP["totals"] = totals(_REP["records"])
    line = summary_line(_REP)
    if line:
        line = f"{line} cold/warm totals {_REP['totals']['total_ms']:.1f}ms "
        line += f"(h2d {_REP['totals']['h2d_calls']} / d2h {_REP['totals']['d2h_calls']})"
    return line


# ---------------------------------------------------------------------------
# self-test: a fake engine with the same call shape (no torch / GPU / evoxc)
# ---------------------------------------------------------------------------


def _self_test():  # pragma: no cover - run as ``python stage_probe.py``
    import types

    def fail(message):
        raise AssertionError(message)

    class FakeCache:
        def __init__(self):
            self.hits = 0
            self.misses = 0

    class FakePopulation:
        pop_size = 4

    class FakeExe:
        pass

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return [self.value]

    def make_engine(default_backend):
        engine = types.ModuleType(f"fake_evoxc_engine_{default_backend}")
        engine.CACHE = FakeCache()
        state = {"built": {}}

        def decode_population(values, types_, sizes, input_len, output_len):
            time.sleep(0.001)
            return FakePopulation()

        def build_cached(fn, spec, program, backend=None, device=None, cache=None, **kwargs):
            del fn, spec, device, kwargs
            time.sleep(0.002)  # simulated trace/compile
            if program in state["built"]:
                cache.hits += 1
                return state["built"][program]
            cache.misses += 1
            exe = FakeExe()
            state["built"][program] = exe
            return exe

        def run(exe, xin, program):
            del exe, xin, program
            time.sleep(0.001)
            return FakeResult(1.0)

        def _run_tree(tree, xin, n_rows, input_len, backend_name, device, options, prepared):
            del n_rows, input_len, device, options
            exe = engine.build_cached(None, None, tree, backend=backend_name, cache=engine.CACHE)
            if backend_name == "numpy":
                return engine.evoxc.run(exe, xin, tree)
            if not prepared:
                prepared.append(engine.evoxc.from_numpy(xin).to("device"))
            result = engine.evoxc.run(exe, prepared[0], tree)
            return result.numpy()

        def evaluate(population, x, *, per_row, out_device=None, backend=None, iree_device_index=None):
            del per_row, out_device, iree_device_index
            prepared = []  # as in the engine: one H2D per call, reused by every tree
            for tree in range(population.pop_size):
                engine._run_tree(tree, x, 4, 2, backend or default_backend, None, {}, prepared)
            time.sleep(0.0005)
            return FakeResult("tensor")

        def sr_fitness(population, inputs, labels, use_MSE=True, **kwargs):
            del use_MSE
            engine.evaluate(population, inputs, per_row=False, **kwargs)
            time.sleep(0.001)
            return FakeResult("loss")

        def from_numpy(value):
            class DeviceTensor:
                def to(self, device):
                    del device
                    return FakeResult(value)

            return DeviceTensor()

        evoxc = types.ModuleType("fake_evoxc")
        evoxc.run = run
        evoxc.from_numpy = from_numpy
        engine.evoxc = evoxc
        engine.decode_population = decode_population
        engine.build_cached = build_cached
        engine._run_tree = _run_tree
        engine.evaluate = evaluate
        engine.sr_fitness = sr_fitness
        engine.cache_stats = lambda: {"hits": engine.CACHE.hits, "misses": engine.CACHE.misses}
        return engine

    def one_call(engine):
        begin()
        population = engine.decode_population(None, None, None, 2, 1)
        engine.sr_fitness(population, "x", "y")
        return snapshot(i=0, outer_ms=50.0)

    for backend in ("numpy", "iree"):
        _reset_install()
        reset()
        engine = make_engine(backend)
        wrapper = install(engine)
        if not wrapper["installed"] or wrapper["missing"]:
            fail(f"[{backend}] install status: {wrapper}")
        container(side="new")

        begin_rep(0, "eval")
        cold = one_call(engine)
        warm = one_call(engine)
        line = finish_rep()

        checks = [
            (cold["decode_calls"] == 1, f"[{backend}] decode_calls {cold['decode_calls']}"),
            (cold["sr_fitness_calls"] == 1, f"[{backend}] sr_fitness_calls"),
            (cold["evaluate_calls"] == 1, f"[{backend}] evaluate_calls"),
            (cold["trees_total"] == 4, f"[{backend}] trees_total {cold['trees_total']}"),
            (cold["trees_invoked"] == 4, f"[{backend}] trees_invoked"),
            (cold["build_calls"] == 4, f"[{backend}] build_calls {cold['build_calls']}"),
            (cold["build_misses"] == 4, f"[{backend}] cold build_misses {cold['build_misses']}"),
            (cold["build_hits"] == 0, f"[{backend}] cold build_hits"),
            (cold["build_miss_ms"] > 0.0, f"[{backend}] cold build_miss_ms"),
            (cold["run_calls"] == 4, f"[{backend}] run_calls {cold['run_calls']}"),
            (cold["readback_calls"] == 4, f"[{backend}] readback_calls"),
            (cold["readback_ms"] > 0.0, f"[{backend}] readback_ms {cold['readback_ms']}"),
            (cold["loss_ms"] > 0.0, f"[{backend}] loss_ms"),
            (warm["build_misses"] == 0, f"[{backend}] warm build_misses"),
            (warm["build_hits"] == 4, f"[{backend}] warm build_hits {warm['build_hits']}"),
            (warm["run_calls"] == 4, f"[{backend}] warm run_calls"),
            (warm["h2d_calls"] == (1 if backend == "iree" else 0), f"[{backend}] h2d {warm['h2d_calls']}"),
            (warm["d2h_calls"] == (4 if backend == "iree" else 0), f"[{backend}] d2h {warm['d2h_calls']}"),
        ]
        for ok, label in checks:
            if not ok:
                fail(label)
        rep = reps()[0]
        t = rep["totals"]
        if not (t["cold_builds"] == 4 and t["warm_misses"] == 0 and t["total_build_misses"] == 4):
            fail(f"[{backend}] totals {t}")
        if t["cold_build_ms"] is None or t["cold_build_ms"] <= 0.0:
            fail(f"[{backend}] cold_build_ms {t['cold_build_ms']}")
        if line is None or "build-miss" not in line:
            fail(f"[{backend}] summary line {line!r}")
        # pass-through: the wrapped build_cached still returns the cached object
        first = engine.build_cached(None, None, "pass-through", cache=engine.CACHE)
        again = engine.build_cached(None, None, "pass-through", cache=engine.CACHE)
        if first is not again:
            fail(f"[{backend}] build_cached must pass the cached executable through")
        print(f"[ok] {backend}: {line}")

    # a degraded engine (no build_cached / no counters) must not raise
    _reset_install()
    reset()
    broken = types.ModuleType("fake_engine_broken")
    broken._run_tree = lambda *a, **k: None
    broken.evaluate = lambda *a, **k: None
    broken.sr_fitness = lambda *a, **k: None
    status_broken = install(broken)
    if status_broken["installed"] or not status_broken["partial"]:
        fail(f"degradation status: {status_broken}")
    if container()["partial"] is not True or not container()["note"]:
        fail("degraded container must be partial with a note")
    print(f"[ok] degraded engine -> partial=True note={status_broken['note']!r}")
    _reset_install()
    reset()
    print("[ok] stage_probe self-test passed")
    return 0


def _reset_install():
    """Drop the install state (test helper; the wrappers stay on their modules)."""
    global _STATUS, _COUNTERS, _RUN_TREE_BACKEND_POS
    _STATUS = None
    _COUNTERS = None
    _RUN_TREE_BACKEND_POS = None


if __name__ == "__main__":
    raise SystemExit(_self_test())
