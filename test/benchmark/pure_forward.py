#!/usr/bin/env python
"""``pure_forward.py`` -- runner core of the pure-forward task (``bench_gp.py --task eval-pure``).

R4 add-on for the side-agnostic harness.  The existing eval tasks (``eval64`` /
``evalfresh``) measure **measurement type 1: the end-to-end public call**
``problem.evaluate(forest)`` -- decode + per-structure ``etl.build_cached`` + per-tree
``etl.run`` + device->host readback + host row assembly + the fitness reduction.  A pure
forward is **measurement type 2: the internal graph forward pass only** -- one invocation =
one full-population forward on a fixed ``x`` with ``per_row=False``, with the results
dropped, so none of the host glue above is in the timed region.  The comparison of the two
is what prices the glue (see ``r4/README.md``; the harness records the end-to-end side of
it in the same process as ``e2e_companion``).

This module owns everything that is engine-facing: the mode/backend resolution, the
frozen-vs-fallback runner construction and the timing core.  ``bench_gp.py`` only parses
flags and writes the payload.  Nothing here imports ``evogp`` / ``etl`` at module import
time (neither side may be forced to have them); the imports happen inside the functions,
i.e. only on the new side and only when a runner is actually built.

Two construction paths, selected by ``runner_source`` (``--pure-runner``):

``frozen``
    the frozen API of the parallel code round --
    ``dense_engine.build_dense_runner(forest, x)`` for the dense mode and
    ``etl_engine.build_specialized_runner(forest, x)`` for the specialized mode (two
    positional arguments: the ``Forest`` and the ``x`` tensor).  When the API is absent on
    this checkout, ``--pure-runner frozen`` fails hard and ``auto`` silently uses the
    fallback; the JSON's ``pure.runner_source`` always says which one ran.
``fallback``
    the local re-implementation below, which replicates exactly what the engine does per
    evaluate call, minus the key derivation, the readback and the row assembly:

    * dense -- ``plan_dense(...)`` (host preprocess + chunk shape, ``per_row=False``),
      ``build_dense(plan)`` (ONE graph for the whole population) and the per-chunk
      invocation arguments, precomputed **once**; the timed loop is one ``etl.run`` per
      chunk and the result is dropped.
    * specialized -- ``decode_population(...)`` once, ``_executable(...)`` once per
      non-constant tree (the shared compile cache, exactly the engine's memo key) and the
      host->device input upload once; the timed loop is one ``etl.run`` per non-constant
      tree and the result is dropped.

Honesty note on what a pure invocation still pays: the ``etl.run`` wrapper's own validation
and the backend invoke are inside the timed loop by design -- they cannot be removed without
measuring nothing.  The compile tax, the input upload, the key derivation, the readback and
the host assembly are all outside it.  Whole-population device-side stacking does not exist
today, so the per-tree ``etl.run`` loop is the honest floor for the specialized route (and
the per-chunk loop for the dense route, which is one call per chunk).

``n_graph_calls_per_run`` is the number of ``etl.run`` calls one invocation makes
(``<= n_trees``: the specialized route skips constant trees, which the engine evaluates as a
constant and never invokes) -- the divisor that turns a per-invocation number into
``per_graph_call_ms``.
"""

import statistics
import time

import numpy as np
import torch

#: Frozen ``pure.runner_source`` labels (the r4 validator recognizes exactly this set).
FROZEN_DENSE = "frozen-api:build_dense_runner"
FROZEN_SPECIALIZED = "frozen-api:build_specialized_runner"
FALLBACK_DENSE = "local-fallback:dense"
FALLBACK_SPECIALIZED = "local-fallback:specialized"
RUNNER_SOURCES = (FROZEN_DENSE, FROZEN_SPECIALIZED, FALLBACK_DENSE, FALLBACK_SPECIALIZED)

#: ``--mode`` values; ``dynamic`` is a documented alias of ``dense``.
MODES = ("auto", "dense", "specialized", "dynamic")
#: ``--pure-runner`` values.
RUNNERS = ("auto", "frozen", "fallback")

#: ``pure.conventions`` -- the one sentence a reader needs to interpret every pure number.
CONVENTIONS = (
    "one invocation = one full-population forward on the fixed x; "
    "no readback/host assembly/key derivation"
)

#: ``e2e_companion.note`` -- what the E2E-minus-pure difference prices.
E2E_COMPANION_NOTE = (
    "E2E minus pure = per-call decode+key derivation + readback + host assembly + "
    "fitness reduction (host-glue + readback overhead)"
)


# ---------------------------------------------------------------------------
# lazy module access (never imports evogp/etl at import time)
# ---------------------------------------------------------------------------


def _etl_engine():
    from evogp.tree import etl_engine

    return etl_engine


def _dense_engine():
    from evogp.tree import dense_engine

    return dense_engine


def resolve_backend_name(explicit=None):
    """Engine backend name (``"numpy"`` / ``"iree"``) for a harness ``--backend`` value.

    ``None`` / ``"default"`` means "set nothing": the engine's own precedence
    (``EVOGP_ETL_BACKEND`` env > built-in default) decides, single-sourced by delegating to
    ``etl_engine._resolve_backend`` (the same helper the engine itself calls).
    """
    arg = None if explicit in (None, "", "default") else str(explicit)
    return _etl_engine()._resolve_backend(arg, None)[0]


def resolve_mode(requested, pop_size, backend_name):
    """Resolve ``--mode`` into the frozen pair ``(mode_requested, mode_resolved)``.

    * ``dense`` / ``specialized`` -- the explicit value wins.
    * ``dynamic`` -- a documented alias of ``dense``; recorded as requested ``dynamic``.
    * ``auto`` -- the engine's own policy, asked rather than duplicated:
      ``dense_engine.resolve_eval_mode()`` reads ``EVOGP_EVAL_MODE`` (an explicit value
      there wins, and is reported) and, when that is still ``auto``,
      ``dense_engine.should_use_dense(pop_size, backend=<resolved backend>)`` answers the
      rest (numpy -> specialized; iree with pop >= 16 -> dense).  This is the same decision
      the wired call sites make, so the pure number and the ``e2e_companion`` number below
      it are on comparable routes.
    """
    requested = str(requested).strip().lower()
    if requested == "dynamic":
        return {"mode_requested": "dynamic", "mode_resolved": "dense"}
    if requested in ("dense", "specialized"):
        return {"mode_requested": requested, "mode_resolved": requested}
    if requested != "auto":
        raise ValueError(f"unsupported mode {requested!r}: expected one of {list(MODES)}")
    dense_engine = _dense_engine()
    env_mode = dense_engine.resolve_eval_mode(None)  # EVOGP_EVAL_MODE (raises on junk)
    if env_mode in ("dense", "specialized"):
        return {"mode_requested": "auto", "mode_resolved": env_mode}
    use_dense = dense_engine.should_use_dense(int(pop_size), backend=backend_name)
    return {
        "mode_requested": "auto",
        "mode_resolved": "dense" if use_dense else "specialized",
    }


def frozen_availability():
    """``{"dense": bool, "specialized": bool}`` -- which frozen runner API exists here.

    Presence-only detection (``getattr``, never a call): the parallel code round lands
    ``dense_engine.build_dense_runner`` / ``etl_engine.build_specialized_runner``, and this
    is how the harness decides between the frozen path and its own fallback.
    """
    return {
        "dense": getattr(_dense_engine(), "build_dense_runner", None) is not None,
        "specialized": getattr(_etl_engine(), "build_specialized_runner", None) is not None,
    }


# ---------------------------------------------------------------------------
# the runner object
# ---------------------------------------------------------------------------


class PureRunner:
    """A constructed pure-forward runner: metadata + one ``invoke()``.

    ``invoke()`` performs ONE full-population forward on the fixed ``x`` and returns
    nothing: the results are deliberately dropped (they are device-resident on iree; nothing
    is read back, and nothing is retained -- holding ``N`` population outputs would only
    grow the allocator).
    """

    def __init__(
        self,
        *,
        source,
        mode,
        backend_name,
        per_row,
        pop_size,
        n_trees,
        n_const_trees,
        n_graph_calls_per_run,
        n_graph_calls_source,
        invoke,
    ):
        self.source = source
        self.mode = mode
        self.backend_name = backend_name
        self.per_row = bool(per_row)
        self.pop_size = int(pop_size)
        self.n_trees = int(n_trees)
        self.n_const_trees = n_const_trees
        self.n_graph_calls_per_run = int(n_graph_calls_per_run)
        self.n_graph_calls_source = n_graph_calls_source
        self._invoke = invoke

    def invoke(self):
        """One full-population forward on the fixed ``x`` (result dropped)."""
        self._invoke()

    def describe(self):
        """The static part of the ``pure`` JSON block (the harness adds the timings)."""
        return {
            "mode_resolved": self.mode,
            "runner_source": self.source,
            "backend": self.backend_name,
            "per_row": self.per_row,
            "pop": self.pop_size,
            "n_trees": self.n_trees,
            "n_const_trees": self.n_const_trees,
            "n_graph_calls_per_run": self.n_graph_calls_per_run,
            "n_graph_calls_source": self.n_graph_calls_source,
        }


def _cache_counters():
    """``(hits, misses)`` of the shared ETL compile cache; ``(None, None)`` when unreadable."""
    try:
        stats = _etl_engine().cache_stats()
        return int(stats["hits"]), int(stats["misses"])
    except Exception:
        return None, None


def _delta(after, before):
    if after is None or before is None:
        return None
    return int(after) - int(before)


def _frozen_attr(obj, *names):
    """First present attribute among ``names`` (frozen-runner introspection helper)."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _frozen_invoke(frozen_runner, x):
    """Adapter for the frozen runner's call form -- see the FROZEN-ADAPTER note.

    FROZEN-ADAPTER: this round's evidence is a zero-argument call (the ``x``, or a
    population-tile of it, was bound at construction).  If the landed API wants the input
    per call instead, adjust only this function (``runner(x)``) -- nothing else depends on
    the call form.
    """
    if callable(frozen_runner):
        return lambda: frozen_runner()
    forward = getattr(frozen_runner, "forward", None)
    if callable(forward):
        return lambda: forward(x)
    raise TypeError(
        "frozen runner of type "
        f"{type(frozen_runner).__name__} is neither callable nor does it expose .forward()"
    )


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_pure_runner(forest, x, *, mode, runner_source="auto", backend=None, tag="pure"):
    """Construct (and time) a pure-forward runner for ``forest`` on the fixed input ``x``.

    Args:
        forest: the population (any ``evogp.tree.Forest``).
        x: the fixed ``(N, input_len)`` input tensor (torch, any device) -- the same object
            the end-to-end path would evaluate on.
        mode: ``"dense"`` or ``"specialized"`` (already resolved by :func:`resolve_mode`).
        runner_source: ``"auto"`` (frozen when available, else the fallback), ``"frozen"``
            (raise when the frozen API is missing) or ``"fallback"`` (force the local
            implementation even when the frozen API exists -- the A/B equivalence check).
        backend: harness ``--backend`` value (``None``/``"default"`` -> the engine's env
            precedence).
        tag: label used in error messages only.

    Returns:
        ``(runner, compile_block)`` where ``runner`` is a :class:`PureRunner` and
        ``compile_block`` is the cold, separate record
        ``{decode_plan_ms, build_ms, construct_total_ms, cache_hits, cache_misses, path}``:
        ``decode_plan_ms`` = host decode/preprocess (population decode or dense plan +
        input conversion, including the one-time device upload), ``build_ms`` = the
        ``build_cached`` calls only, ``construct_total_ms`` = the whole construction
        (so the residual is the invocation-argument precompute).  ``decode_plan_ms`` /
        ``build_ms`` are ``None`` on the frozen path (the API does the whole construction in
        one call, so only the total is separable).

    The whole construction happens **before** any timed invocation, and it fixes the shapes
    the compile key bakes in (``n_rows``, ``input_len``, ``L``, ``per_row``, backend,
    device), so the timed loop can never trigger a rebuild.
    """
    mode = str(mode).strip().lower()
    if mode not in ("dense", "specialized"):
        raise ValueError(f"mode must be dense|specialized, got {mode!r} (see resolve_mode)")
    runner_source = str(runner_source).strip().lower()
    if runner_source not in RUNNERS:
        raise ValueError(f"runner_source must be one of {list(RUNNERS)}, got {runner_source!r}")
    if int(getattr(x, "ndim", 0)) != 2:
        raise ValueError(f"x must be 2-D (rows, input_len), got shape {getattr(x, 'shape', None)}")
    backend_name = resolve_backend_name(backend)
    available = frozen_availability()
    if runner_source == "auto":
        runner_source = "frozen" if available[mode] else "fallback"
    if runner_source == "frozen" and not available[mode]:
        raise RuntimeError(
            f"[{tag}] --pure-runner frozen: the frozen pure-forward API for mode={mode} is "
            "missing in this checkout (evogp.tree.dense_engine.build_dense_runner / "
            "evogp.tree.etl_engine.build_specialized_runner); use --pure-runner auto|fallback"
        )
    if runner_source == "frozen":
        return _construct_frozen(forest, x, mode=mode, backend_name=backend_name)
    return _construct_fallback(forest, x, mode=mode, backend_name=backend_name)


def _construct_frozen(forest, x, *, mode, backend_name):
    """Build through the frozen API (``build_{dense,specialized}_runner(forest, x)``)."""
    etl_engine = _etl_engine()
    dense_engine = _dense_engine()
    if mode == "dense":
        build_fn = dense_engine.build_dense_runner  # FROZEN-ADAPTER (call form)
        source = FROZEN_DENSE
    else:
        build_fn = etl_engine.build_specialized_runner  # FROZEN-ADAPTER (call form)
        source = FROZEN_SPECIALIZED
    hits_before, misses_before = _cache_counters()
    start = time.perf_counter()
    frozen_runner = build_fn(forest, x)
    construct_total_ms = (time.perf_counter() - start) * 1000.0
    hits_after, misses_after = _cache_counters()

    n_trees = _frozen_attr(frozen_runner, "n_trees", "pop_size")
    n_const_trees = _frozen_attr(frozen_runner, "n_const_trees")
    n_graph_calls = _frozen_attr(frozen_runner, "n_graph_calls_per_run", "n_graph_calls")
    if n_graph_calls is None:
        # No graph-call counter on the frozen object: the cheap honest choice is 1 (a single
        # fused population call), recorded as an assumption, never as a measurement.
        n_graph_calls, n_graph_calls_source = 1, "assumed-1"
    else:
        n_graph_calls_source = "introspected"
    if n_graph_calls < 1:
        raise RuntimeError(
            f"frozen runner reports n_graph_calls_per_run={n_graph_calls}: a pure invocation "
            "must make at least one graph call"
        )
    runner = PureRunner(
        source=source,
        mode=mode,
        backend_name=backend_name,
        per_row=False,
        pop_size=int(forest.pop_size),
        n_trees=int(forest.pop_size) if n_trees is None else int(n_trees),
        n_const_trees=None if n_const_trees is None else int(n_const_trees),
        n_graph_calls_per_run=int(n_graph_calls),
        n_graph_calls_source=n_graph_calls_source,
        invoke=_frozen_invoke(frozen_runner, x),
    )
    return runner, {
        "decode_plan_ms": None,
        "build_ms": None,
        "construct_total_ms": construct_total_ms,
        "cache_hits": _delta(hits_after, hits_before),
        "cache_misses": _delta(misses_after, misses_before),
        "path": "frozen",
    }


def _construct_fallback(forest, x, *, mode, backend_name):
    """Build through the local fallback (the engine's own per-call steps, minus the glue)."""
    if mode == "dense":
        return _fallback_dense(forest, x, backend_name=backend_name)
    return _fallback_specialized(forest, x, backend_name=backend_name)


def _fallback_dense(forest, x, *, backend_name):
    """Fallback dense runner: ONE graph for the population, one ``etl.run`` per chunk."""
    import etl  # lazy: only the new side reaches this function

    etl_engine = _etl_engine()
    dense_engine = _dense_engine()
    n_rows = int(x.shape[0])
    hits_before, misses_before = _cache_counters()
    start = time.perf_counter()
    plan = dense_engine.plan_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        int(forest.input_len),
        int(forest.output_len),
        n_rows,
        per_row=False,
        backend=backend_name,
    )
    decode_plan_end = time.perf_counter()
    executable = dense_engine.build_dense(plan, backend=backend_name)
    build_end = time.perf_counter()
    device = etl_engine._resolve_backend(backend_name, None)[1]
    to_device = backend_name != "numpy"
    # The engine reads x back once per evaluate call; here it is read ONCE for the whole
    # task (a fixed x) and, on a device backend, uploaded once -- both outside the timed loop.
    # ``per_row=False`` is fixed for this task, so every chunk consumes data rows.
    host_x = etl_engine._input_to_numpy(x, plan.input_len, "x")
    args_per_chunk = []
    for p0 in range(0, plan.pop_size, plan.p_chunk):
        columns = dense_engine._prep_columns(plan.prep, p0, plan.p_chunk)
        for chunk_index in range(plan.n_chunks):
            n0 = chunk_index * plan.n_chunk
            n_chunk = min(plan.n_chunk, plan.n_rows - n0)
            xin = np.ascontiguousarray(host_x[n0 : n0 + n_chunk])
            xin = dense_engine._pad_rows(xin, plan.n_chunk)
            if to_device:
                args_per_chunk.append(
                    tuple(etl.from_numpy(a).to(device) for a in columns)
                    + (etl.from_numpy(xin).to(device), plan.cfg)
                )
            else:
                args_per_chunk.append(columns + (xin, plan.cfg))
    construct_total_ms = (time.perf_counter() - start) * 1000.0
    hits_after, misses_after = _cache_counters()
    quiet = etl_engine._quiet

    def invoke():
        for args in args_per_chunk:
            with quiet():
                etl.run(executable, *args)

    runner = PureRunner(
        source=FALLBACK_DENSE,
        mode="dense",
        backend_name=backend_name,
        per_row=False,
        pop_size=plan.pop_size,
        # The dense graph evaluates every row, constant trees included, so no tree is
        # skipped by construction (the key is a dense-route convention, see r4/README.md).
        n_trees=plan.pop_size,
        n_const_trees=0,
        n_graph_calls_per_run=len(args_per_chunk),
        n_graph_calls_source="measured",
        invoke=invoke,
    )
    return runner, {
        "decode_plan_ms": (decode_plan_end - start) * 1000.0,
        "build_ms": (build_end - decode_plan_end) * 1000.0,
        "construct_total_ms": construct_total_ms,
        "cache_hits": _delta(hits_after, hits_before),
        "cache_misses": _delta(misses_after, misses_before),
        "path": "fallback",
    }


def _fallback_specialized(forest, x, *, backend_name):
    """Fallback specialized runner: the engine's decode + per-tree ``etl.run`` loop."""
    import etl  # lazy: only the new side reaches this function

    etl_engine = _etl_engine()
    n_rows = int(x.shape[0])
    hits_before, misses_before = _cache_counters()
    start = time.perf_counter()
    population = etl_engine.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        int(forest.input_len),
        int(forest.output_len),
    )
    # Host input conversion + the one-time device upload (the engine memoizes one upload per
    # evaluate call; with a fixed x this task uploads once for the whole task).
    host_x = etl_engine._input_to_numpy(x, population.input_len, "x")
    backend_resolved, device, options = etl_engine._resolve_backend(backend_name, None)
    dev_x = None if backend_resolved == "numpy" else etl.from_numpy(host_x).to(device)
    decode_plan_end = time.perf_counter()
    built = [
        (
            tree.program,
            etl_engine._executable(
                tree.program, n_rows, population.input_len, backend_resolved, device, options
            ),
        )
        for tree in population.trees
        if tree.program is not None
    ]
    build_end = time.perf_counter()
    hits_after, misses_after = _cache_counters()
    quiet = etl_engine._quiet
    input_arg = dev_x if dev_x is not None else host_x

    def invoke():
        for program, executable in built:
            with quiet():
                etl.run(executable, input_arg, program)

    runner = PureRunner(
        source=FALLBACK_SPECIALIZED,
        mode="specialized",
        backend_name=backend_resolved,
        per_row=False,
        pop_size=population.pop_size,
        n_trees=population.pop_size,
        # A constant/collapsed tree is evaluated as a constant by the engine: it is never
        # invoked, so it contributes no graph call to the loop above.
        n_const_trees=population.pop_size - len(built),
        n_graph_calls_per_run=len(built),
        n_graph_calls_source="measured",
        invoke=invoke,
    )
    return runner, {
        "decode_plan_ms": (decode_plan_end - start) * 1000.0,
        "build_ms": (build_end - decode_plan_end) * 1000.0,
        "construct_total_ms": (time.perf_counter() - start) * 1000.0,
        "cache_hits": _delta(hits_after, hits_before),
        "cache_misses": _delta(misses_after, misses_before),
        "path": "fallback",
    }


# ---------------------------------------------------------------------------
# timing core
# ---------------------------------------------------------------------------


def time_block(runner, invocations, *, sync=None, sync_each=False):
    """Time ``invocations`` invocations of ``runner``; returns the block's wall time in ms.

    ``sync_each=False`` (the default the harness uses): one ``torch.cuda.synchronize()``
    pair around the whole block, so ``block_ms / N`` is the per-invocation cost only when
    the invoke path is synchronous.  The iree path was verified against a per-invocation
    control in the R4 smoke; ``sync_each=True`` is that control (it costs one sync per
    invocation, which the default deliberately does not charge to the runner).
    """
    sync = torch.cuda.synchronize if sync is None else sync
    count = int(invocations)
    if count < 1:
        raise ValueError(f"invocations must be >= 1, got {invocations}")
    sync()
    start = time.perf_counter()
    if sync_each:
        for _ in range(count):
            runner.invoke()
            sync()
    else:
        for _ in range(count):
            runner.invoke()
        sync()
    return (time.perf_counter() - start) * 1000.0


def summarize(per_invocation_ms, *, n_graph_calls_per_run, n_trees):
    """Per-rep per-invocation numbers -> the ``pure`` block's summary fields.

    ``per_population_evaluate_ms`` IS the median per-invocation number by definition: one
    invocation is one full-population forward on the fixed x, which is exactly one
    population evaluate minus the readback/assembly (see ``r4/README.md`` glossary).
    ``per_graph_call_ms`` divides that by ``n_graph_calls_per_run`` and ``per_tree_us``
    normalizes by the population size -- both are convenience normalizations of the same
    median, never independent measurements.
    """
    values = [float(v) for v in per_invocation_ms if isinstance(v, (int, float))]
    if not values:
        return {
            "median_per_invocation_ms": None,
            "min_per_invocation_ms": None,
            "mean_per_invocation_ms": None,
            "per_graph_call_ms": None,
            "per_tree_us": None,
            "per_population_evaluate_ms": None,
        }
    median = statistics.median(values)
    calls = max(1, int(n_graph_calls_per_run or 1))
    trees = max(1, int(n_trees or 1))
    return {
        "median_per_invocation_ms": median,
        "min_per_invocation_ms": min(values),
        "mean_per_invocation_ms": statistics.fmean(values),
        "per_graph_call_ms": median / calls,
        "per_tree_us": median * 1000.0 / trees,
        "per_population_evaluate_ms": median,
    }


def time_e2e_companion(problem, forest, reps, *, sync=None):
    """K synchronized ``problem.evaluate(forest)`` calls -- the end-to-end companion.

    Runs in the SAME process as the pure measurement, with the compile cache already warm
    from the pure construction, so ``per_call_ms[0]`` is a warm first call and the difference
    to the pure median is the per-call host cost (decode + key derivation + readback + row
    assembly + the fitness reduction), not a compile.
    """
    sync = torch.cuda.synchronize if sync is None else sync
    count = int(reps)
    if count < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    per_call_ms = []
    for _ in range(count):
        sync()
        start = time.perf_counter()
        problem.evaluate(forest)
        sync()
        per_call_ms.append((time.perf_counter() - start) * 1000.0)
    warm = per_call_ms[1:]
    # With K == 1 there is no warm call: report the single call as the median (documented in
    # r4/README.md) instead of a null the validator would have to special-case.
    median_warm = statistics.median(warm) if len(warm) >= 2 else statistics.median(per_call_ms)
    return {
        "eval_reps": count,
        "per_call_ms": per_call_ms,
        "first_call_ms": per_call_ms[0],
        "median_warm_ms": median_warm,
        "note": E2E_COMPANION_NOTE,
    }


__all__ = [
    "FROZEN_DENSE",
    "FROZEN_SPECIALIZED",
    "FALLBACK_DENSE",
    "FALLBACK_SPECIALIZED",
    "RUNNER_SOURCES",
    "MODES",
    "RUNNERS",
    "CONVENTIONS",
    "E2E_COMPANION_NOTE",
    "PureRunner",
    "build_pure_runner",
    "frozen_availability",
    "resolve_backend_name",
    "resolve_mode",
    "summarize",
    "time_block",
    "time_e2e_companion",
]
