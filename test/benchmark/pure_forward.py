#!/usr/bin/env python
"""``pure_forward.py`` -- runner core of the pure-forward task (``bench_gp.py --task eval-pure``).

R4 add-on for the side-agnostic harness.  The existing eval tasks (``eval64`` /
``evalfresh``) measure **measurement type 1: the end-to-end public call**
``problem.evaluate(forest)`` -- decode + per-structure ``evoxc.build_cached`` + per-tree
``evoxc.run`` + device->host readback + host row assembly + the fitness reduction.  A pure
forward is **measurement type 2: the internal graph forward pass only** -- one invocation =
one full-population forward on a fixed ``x`` with ``per_row=False``, with the results
dropped, so none of the host glue above is in the timed region.  The comparison of the two
is what prices the glue (see ``r4/README.md``; the harness records the end-to-end side of
it in the same process as ``e2e_companion``).

This module owns everything that is engine-facing: the mode/backend resolution, the
frozen-vs-fallback runner construction and the timing core.  ``bench_gp.py`` only parses
flags and writes the payload.  Nothing here imports ``evogp`` / ``evoxc`` at module import
time (neither side may be forced to have them); the imports happen inside the functions,
i.e. only on the new side and only when a runner is actually built.

Two construction paths, selected by ``runner_source`` (``--pure-runner``):

``frozen``
    the frozen API of the parallel code round --
    ``dense_engine.build_dense_runner(forest, x)`` for the dense mode and
    ``evoxc_engine.build_specialized_runner(forest, x)`` for the specialized mode.  The
    validated call forms (``src/evogp/tree/RUNNER_HOOKS.md`` §1/§3, both builders landed in
    this checkout) are **two positional arguments at build** (the ``Forest`` and the ``x``
    tensor -- ``per_row=False``, the backend from the environment) and a **no-arg
    ``runner.run()``** at invoke time, returning the raw ``evoxc`` result(s): one
    ``(pop_size, n_rows)`` tensor for dense, a ``pop_size``-long tuple for specialized
    (``None`` at a constant tree).  When the API is absent on this checkout,
    ``--pure-runner frozen`` fails hard and ``auto`` silently uses the fallback; the JSON's
    ``pure.runner_source`` always says which one ran.
``fallback``
    the local re-implementation below, which replicates exactly what the engine does per
    evaluate call, minus the key derivation, the readback and the row assembly:

    * dense -- ``plan_dense(...)`` (host preprocess + chunk shape, ``per_row=False``),
      ``build_dense(plan)`` (ONE graph for the whole population) and the per-chunk
      invocation arguments, precomputed **once**; the timed loop is one ``evoxc.run`` per
      chunk and the result is dropped.
    * specialized -- ``decode_population(...)`` once, ``_executable(...)`` once per
      non-constant tree (the shared compile cache, exactly the engine's memo key) and the
      host->device input upload once; the timed loop is one ``evoxc.run`` per non-constant
      tree and the result is dropped.

Honesty note on what a pure invocation still pays: the ``evoxc.run`` wrapper's own validation
and the backend invoke are inside the timed loop by design -- they cannot be removed without
measuring nothing.  The compile tax, the input upload, the key derivation, the readback and
the host assembly are all outside it.  Whole-population device-side stacking does not exist
today, so the per-tree ``evoxc.run`` loop is the honest floor for the specialized route (and
the per-chunk loop for the dense route, which is one call per chunk).

``n_graph_calls_per_run`` is the number of ``evoxc.run`` calls one invocation makes
(``<= n_trees``: the specialized route skips constant trees, which the engine evaluates as a
constant and never invokes) -- the divisor that turns a per-invocation number into
``per_graph_call_ms``.

``capture_outputs()`` (:func:`PureRunner.capture_outputs`, wired by ``bench_gp.py
--output-checksum`` / ``--save-output-dir``) is the **out-of-band** counterpart of an
invocation: it performs ONE further invocation and materializes the raw results on the host
(``numpy`` backend: the result is already host-backed; device backend: an explicit
``to(Device("cpu", 0))`` copy) as float32 C-contiguous blocks -- dense: one
``(pop_size, n_rows)`` block; specialized: one ``(n_rows, output_len)`` block per
NON-constant tree, in population order.  :func:`digest_blocks` then frames those blocks
deterministically into a sha256 (plus shape/dtype/NaN/+-inf census).  None of this is ever
timed: the harness captures after the last timed block, and the readback is a consumer-side
copy by definition.
"""

import hashlib
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

#: The canonical digital-digest framing of :func:`digest_blocks` -- one frame per block, in
#: order: this prefix (ndim, shape, ``dtype.str``, byte count) followed by the C-order bytes.
DIGEST_FRAMING = "b'block:<ndim>:<d0,d1,...>:<dtype.str>:<nbytes>:' + block.tobytes(order='C')"

#: ``pure.output_checksum.note`` -- what the captured digest is (and is not) for a reader.
OUTPUT_CHECKSUM_NOTE = (
    "one out-of-band capture_outputs() invocation after ALL timing (the reps and the e2e "
    "companion): an extra invocation plus a consumer-side host readback, never part of any "
    f"timed number; sha256 frames one block per entry in order as {DIGEST_FRAMING}"
)

# ---------------------------------------------------------------------------
# lazy module access (never imports evogp/evoxc at import time)
# ---------------------------------------------------------------------------


def _evoxc_engine():
    from evogp.tree import evoxc_engine

    return evoxc_engine


def _dense_engine():
    from evogp.tree import dense_engine

    return dense_engine


def resolve_backend_name(explicit=None):
    """Engine backend name (``"numpy"`` / ``"iree"``) for a harness ``--backend`` value.

    ``None`` / ``"default"`` means "set nothing": the engine's own precedence
    (``EVOGP_EVOXC_BACKEND`` env > built-in default) decides, single-sourced by delegating to
    ``evoxc_engine._resolve_backend`` (the same helper the engine itself calls).
    """
    arg = None if explicit in (None, "", "default") else str(explicit)
    return _evoxc_engine()._resolve_backend(arg, None)[0]


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
    ``dense_engine.build_dense_runner`` / ``evoxc_engine.build_specialized_runner``, and this
    is how the harness decides between the frozen path and its own fallback.
    """
    return {
        "dense": getattr(_dense_engine(), "build_dense_runner", None) is not None,
        "specialized": getattr(_evoxc_engine(), "build_specialized_runner", None) is not None,
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

    ``capture_outputs()`` is the deliberate exception, and it is **out of band**: one further
    invocation whose raw results ARE materialized on the host (a consumer-side readback) so a
    harness can fingerprint the numbers without contaminating a timed block.  See
    :func:`digest_blocks`.
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
        capture=None,
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
        self._capture = capture

    def invoke(self):
        """One full-population forward on the fixed ``x`` (result dropped)."""
        self._invoke()

    def capture_outputs(self):
        """ONE extra invocation, read back to host -- out of band, NEVER timed.

        Returns the captured host blocks (float32, C-contiguous): dense -> a one-element
        list holding the ``(pop_size, n_rows)`` block; specialized -> one
        ``(n_rows, output_len)`` block per NON-constant tree, in population order.

        This is the only method that reads a result back, and it pays exactly one extra
        invocation plus the readback: call it after the last timed block (which is what
        ``bench_gp.py --output-checksum`` does).  Errors are deliberately not caught -- a
        failed capture is a failed run, never a silently empty digest.
        """
        if self._capture is None:
            raise RuntimeError(
                "this PureRunner carries no capture callable (constructed without one)"
            )
        return self._capture()

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
    """``(hits, misses)`` of the shared evoxc compile cache; ``(None, None)`` when unreadable."""
    try:
        stats = _evoxc_engine().cache_stats()
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


def _frozen_call(frozen_runner, x):
    """The frozen runner's call form as a zero-arg callable RETURNING the raw result.

    FROZEN-ADAPTER: the validated forms at this commit are ``build_dense_runner(forest, x)``
    / ``build_specialized_runner(forest, x)`` (two positional arguments -- the input is bound
    at build, ``per_row=False``, backend from the environment) and a **no-arg**
    ``runner.run()`` returning the raw ``evoxc`` result(s); see
    ``src/evogp/tree/RUNNER_HOOKS.md`` §1/§3 (and §3's own recorded ``.run()`` numbers) for
    the contract this was checked against.  The two older guesses are kept behind it, in
    order, so the adapter stays honest against a foreign checkout: a bare callable, then
    ``.run()``, then ``.forward(x)``.  Change ONLY this function when a call form changes --
    nothing else depends on it.
    """
    if callable(frozen_runner):
        return lambda: frozen_runner()
    run = getattr(frozen_runner, "run", None)
    if callable(run):
        return lambda: run()
    forward = getattr(frozen_runner, "forward", None)
    if callable(forward):
        return lambda: forward(x)
    raise TypeError(
        "frozen runner of type "
        f"{type(frozen_runner).__name__} is neither callable nor does it expose "
        ".run()/.forward()"
    )


def _frozen_invoke(frozen_runner, x):
    """The timed invocation on the frozen runner (result deliberately dropped).

    Same validated call form as :func:`_frozen_call` -- a missing one raises ``TypeError``
    here, at build time, never inside the timed loop.
    """
    call = _frozen_call(frozen_runner, x)
    return lambda: call()


def to_host_block(raw, backend_name):
    """One raw ``evoxc`` result -> host float32 C-contiguous numpy (an out-of-band readback).

    ``numpy`` backend: the result never left the host, so ``.numpy()`` is a view/copy of the
    runner's own memory.  Device backend: an explicit ``to(Device("cpu", 0))`` copy, i.e. a
    consumer-side readback -- exactly the hop no pure number pays for.  Called only from the
    capture path, never from a timed block.
    """
    import evoxc  # lazy: only the new side reaches this function

    if backend_name == "numpy":
        host = raw.numpy()
    else:
        host = raw.to(evoxc.core.Device("cpu", 0)).numpy()
    return np.ascontiguousarray(host, dtype=np.float32)


def _capture_blocks(mode, raw, backend_name):
    """Raw runner result(s) -> captured host blocks (dense: one; specialized: non-constants).

    ``mode`` is the resolved route: ``specialized`` returns a ``pop_size``-long tuple whose
    ``None`` entries are the constant trees (no graph was compiled, so there is nothing to
    read back), and only the real results become blocks.
    """
    if mode == "dense":
        return [to_host_block(raw, backend_name)]
    return [to_host_block(entry, backend_name) for entry in raw if entry is not None]


def digest_blocks(blocks):
    """Deterministic digest of captured host blocks (out-of-band evidence, never timed).

    Every block is normalized to float32 C-order first, then hashed through the canonical
    framing :data:`DIGEST_FRAMING` -- one frame per block, in order, so the same values with
    a different shape, dtype or trailing padding cannot collide by accident.

    Returns ``{"sha256", "n_blocks", "shapes", "dtypes", "total_elems", "nan_count",
    "inf_count"}``: the sha256 (64 hex chars) of the framed concatenation, the block count,
    the per-block shapes (``[[d0, d1, ...], ...]``) and ``dtype.str`` names, the element
    total, and the NaN / +-inf census of the same host blocks (worth recording here because
    the engine's ``-inf`` NaN-replacement semantics make both real).
    """
    hasher = hashlib.sha256()
    shapes = []
    dtypes = []
    total_elems = 0
    nan_count = 0
    inf_count = 0
    for block in blocks:
        array = np.ascontiguousarray(block, dtype=np.float32)
        shape = [int(dim) for dim in array.shape]
        hasher.update(
            "block:"
            f"{array.ndim}:"
            f"{','.join(str(dim) for dim in shape)}:"
            f"{array.dtype.str}:"
            f"{array.nbytes}:".encode()
        )
        hasher.update(array.tobytes(order="C"))
        shapes.append(shape)
        dtypes.append(array.dtype.str)
        total_elems += int(array.size)
        if array.size:
            nan_count += int(np.count_nonzero(np.isnan(array)))
            inf_count += int(np.count_nonzero(np.isinf(array)))
    return {
        "sha256": hasher.hexdigest(),
        "n_blocks": len(blocks),
        "shapes": shapes,
        "dtypes": dtypes,
        "total_elems": total_elems,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

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
            "evogp.tree.evoxc_engine.build_specialized_runner); use --pure-runner auto|fallback"
        )
    if runner_source == "frozen":
        return _construct_frozen(forest, x, mode=mode, backend_name=backend_name)
    return _construct_fallback(forest, x, mode=mode, backend_name=backend_name)


def _construct_frozen(forest, x, *, mode, backend_name):
    """Build through the frozen API (``build_{dense,specialized}_runner(forest, x)``)."""
    evoxc_engine = _evoxc_engine()
    dense_engine = _dense_engine()
    if mode == "dense":
        build_fn = dense_engine.build_dense_runner  # FROZEN-ADAPTER (call form)
        source = FROZEN_DENSE
    else:
        build_fn = evoxc_engine.build_specialized_runner  # FROZEN-ADAPTER (call form)
        source = FROZEN_SPECIALIZED
    hits_before, misses_before = _cache_counters()
    start = time.perf_counter()
    frozen_runner = build_fn(forest, x)
    construct_total_ms = (time.perf_counter() - start) * 1000.0
    hits_after, misses_after = _cache_counters()

    n_trees = _frozen_attr(frozen_runner, "n_trees", "pop_size")
    # The landed runners name the counter ``invocations`` (dense: 1; specialized: the number
    # of non-constant trees); the older candidate names stay ahead of it for a foreign
    # checkout, and only a runner without ANY counter falls back to the assumed 1.
    n_graph_calls = _frozen_attr(
        frozen_runner, "n_graph_calls_per_run", "n_graph_calls", "invocations"
    )
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
    # The landed runners report the constant trees as rows instead of as a count:
    # ``constant_rows`` is ``pop_size``-long with ``None`` at every non-constant tree
    # (specialized only).  The dense route evaluates constant trees like any other, so it
    # has zero skipped trees by construction -- the same convention the fallback uses.
    constant_rows = _frozen_attr(frozen_runner, "constant_rows")
    if constant_rows is not None:
        n_const_trees = sum(1 for row in constant_rows if row is not None)
    elif mode == "dense":
        n_const_trees = 0
    else:
        n_const_trees = _frozen_attr(frozen_runner, "n_const_trees")
    # The runner resolved the backend itself (``backend=None`` -> the environment, which
    # ``bench_gp.py`` sets from ``--backend``); prefer its own answer when it has one.
    runner_backend = _frozen_attr(frozen_runner, "backend_name") or backend_name
    call = _frozen_call(frozen_runner, x)  # FROZEN-ADAPTER: raises TypeError if unsupported

    def invoke():
        call()  # the raw result is deliberately dropped (see PureRunner.invoke)

    def capture():
        return _capture_blocks(mode, call(), runner_backend)

    runner = PureRunner(
        source=source,
        mode=mode,
        backend_name=runner_backend,
        per_row=False,
        pop_size=int(forest.pop_size),
        n_trees=int(forest.pop_size) if n_trees is None else int(n_trees),
        n_const_trees=None if n_const_trees is None else int(n_const_trees),
        n_graph_calls_per_run=int(n_graph_calls),
        n_graph_calls_source=n_graph_calls_source,
        invoke=invoke,
        capture=capture,
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
    """Fallback dense runner: ONE graph for the population, one ``evoxc.run`` per chunk."""
    import evoxc  # lazy: only the new side reaches this function

    evoxc_engine = _evoxc_engine()
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
    device = evoxc_engine._resolve_backend(backend_name, None)[1]
    to_device = backend_name != "numpy"
    # The engine reads x back once per evaluate call; here it is read ONCE for the whole
    # task (a fixed x) and, on a device backend, uploaded once -- both outside the timed loop.
    # ``per_row=False`` is fixed for this task, so every chunk consumes data rows.
    host_x = evoxc_engine._input_to_numpy(x, plan.input_len, "x")
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
                    tuple(evoxc.from_numpy(a).to(device) for a in columns)
                    + (evoxc.from_numpy(xin).to(device), plan.cfg)
                )
            else:
                args_per_chunk.append(columns + (xin, plan.cfg))
    construct_total_ms = (time.perf_counter() - start) * 1000.0
    hits_after, misses_after = _cache_counters()
    quiet = evoxc_engine._quiet

    def invoke():
        for args in args_per_chunk:
            with quiet():
                evoxc.run(executable, *args)

    def capture():
        # Same calls, same order, same args as ``invoke`` -- only the results are kept, then
        # read back to the host.  Out of band by construction (see PureRunner.capture_outputs).
        blocks = []
        for args in args_per_chunk:
            with quiet():
                raw = evoxc.run(executable, *args)
            blocks.append(to_host_block(raw, backend_name))
        return blocks

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
        capture=capture,
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
    """Fallback specialized runner: the engine's decode + per-tree ``evoxc.run`` loop."""
    import evoxc  # lazy: only the new side reaches this function

    evoxc_engine = _evoxc_engine()
    n_rows = int(x.shape[0])
    hits_before, misses_before = _cache_counters()
    start = time.perf_counter()
    population = evoxc_engine.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        int(forest.input_len),
        int(forest.output_len),
    )
    # Host input conversion + the one-time device upload (the engine memoizes one upload per
    # evaluate call; with a fixed x this task uploads once for the whole task).
    host_x = evoxc_engine._input_to_numpy(x, population.input_len, "x")
    backend_resolved, device, options = evoxc_engine._resolve_backend(backend_name, None)
    dev_x = None if backend_resolved == "numpy" else evoxc.from_numpy(host_x).to(device)
    decode_plan_end = time.perf_counter()
    built = [
        (
            tree.program,
            evoxc_engine._executable(
                tree.program, n_rows, population.input_len, backend_resolved, device, options
            ),
        )
        for tree in population.trees
        if tree.program is not None
    ]
    build_end = time.perf_counter()
    hits_after, misses_after = _cache_counters()
    quiet = evoxc_engine._quiet
    input_arg = dev_x if dev_x is not None else host_x

    def invoke():
        for program, executable in built:
            with quiet():
                evoxc.run(executable, input_arg, program)

    def capture():
        # Same calls, same order, same args as ``invoke`` -- only the results are kept.  The
        # constant trees are not in ``built``, so they contribute no block (matching the
        # dense/specialized block convention documented in PureRunner.capture_outputs).
        blocks = []
        for program, executable in built:
            with quiet():
                raw = evoxc.run(executable, input_arg, program)
            blocks.append(to_host_block(raw, backend_resolved))
        return blocks

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
        capture=capture,
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
    "DIGEST_FRAMING",
    "OUTPUT_CHECKSUM_NOTE",
    "PureRunner",
    "build_pure_runner",
    "digest_blocks",
    "frozen_availability",
    "resolve_backend_name",
    "resolve_mode",
    "summarize",
    "time_block",
    "time_e2e_companion",
    "to_host_block",
]
