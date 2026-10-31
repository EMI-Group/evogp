"""One-graph "dense" population evaluator for the evoxc engine (integrated).

This module evaluates a whole population with ONE compiled evoxc executable fed
with the population's node tensors as RUNTIME DATA, instead of one compiled
graph per tree structure.  The specialized engine
(:mod:`evogp.tree.evoxc_engine`) compiles one evoxc graph per *tree structure* and
calls it once per tree: a ``pop_size`` fitness pass is ``pop_size``
``build_cached`` lookups + ``pop_size`` ``evoxc.run`` calls + ``pop_size`` device
readbacks (IREE), and a new structure costs a 0.4-2.6 s compile there.  The dense
path compiles ONE graph per *shape signature*
``(L, P, N, input_len, per_row, update_mode, backend, device, options)`` and
feeds the decoded population through ordinary tensor inputs, so the compile is
reused across calls, generations and populations of the same shape.

Where the dense path is used (it is wired into the public API)
--------------------------------------------------------------
Five call sites try :func:`evaluate_dense` / :func:`sr_fitness_dense` first and
fall back to the specialized engine; the raw population tensors are already on
hand there, so the dense path never runs the host decode:

============================= ========================== ===================
call site                     per_row                    dense entry point
============================= ========================== ===================
``Forest.forward``            ``True``                   :func:`evaluate_dense`
``Forest.batch_forward``      ``False``                  :func:`evaluate_dense`
``Forest.SR_fitness``         ``False``                  :func:`sr_fitness_dense`
``Tree.forward``              1-D x -> True, 2-D -> False :func:`evaluate_dense`
``Tree.SR_fitness``           ``False``                  :func:`sr_fitness_dense`
============================= ========================== ===================

The **fallback is silent but deliberate**: a :class:`DenseUnsupported` (a
multi-output population, a tree longer than the trace cap, structural doubt)
re-runs the very same call on the existing ``decode_population`` +
``evoxc_engine.evaluate`` / ``evoxc_engine.sr_fitness`` path, which owns the error
contract for malformed trees.  The ONE exception is an over-cap tree on a
non-numpy backend, where the fallback (``pop_size`` per-structure compiles)
really is the expensive outcome: that decline emits a deduped once-per-process
:class:`RuntimeWarning` naming the longest tree, the cap and the env var
(:func:`_warn_trace_cap`) and then falls back exactly like the others.  Nothing
else is caught -- every other exception propagates, and the result contracts are
identical either way (fresh, contiguous, ``requires_grad=False`` float32; SR
fitness returns a fresh writable ``(pop_size,)`` positive-loss tensor).

Strategy switch: ``EVOGP_EVAL_MODE``
------------------------------------
This module is the ONE route dispatcher for the library's four evaluation
engines: ``specialized`` (:mod:`evogp.tree.evoxc_engine`, one graph per tree
structure), ``dense`` (this module, one all-ops graph per shape signature),
``batch`` (:mod:`evogp.tree.batch_engine`, ONE merged graph for the whole
population -- opt-in only, see its bullet below) and ``fused``
(:mod:`evogp.tree.fused_engine`, one Triton launch per population -- opt-in only,
see its bullet below).
One master switch, read **per call** (exactly like ``EVOGP_EVOXC_BACKEND`` in
``evoxc_engine._resolve_backend``), case-insensitive, default ``auto``:

``auto`` (default)
    the measured policy, implemented once in :func:`resolve_route` (the fused half
    in :func:`auto_fused_decline`, the rows half in :func:`auto_rows_decline`),
    which also decides the ENGINE and the BACKEND per call.  With no explicit
    backend (neither ``EVOGP_EVOXC_BACKEND`` nor ``backend=``) and a GPU usable
    from torch, ``auto`` runs, in this order:

    1. the FUSED (Triton) route (:mod:`evogp.tree.fused_engine`) from
       ``pop_size >= EVOGP_FUSED_MIN_POP`` (default 16,
       :func:`resolve_fused_min_pop`) up, while ``triton`` is importable AND the
       population fits that route's memory envelope.  It is the DEFAULT because it
       is the fastest measured route -- ONE Triton launch for the whole
       population, no evoxc graph, no host decode, no readback on the value path,
       no per-shape compile: 8.6x-663x the dense route on the measured cells
       (:mod:`evogp.tree.fused_engine`, ``FUSED.md``);
    2. else the DENSE route on ``iree`` from ``pop_size >=
       EVOGP_DENSE_AUTO_MIN_POP`` (default :data:`DEFAULT_AUTO_MIN_POP` = 64) up;
    3. else the SPECIALIZED route on ``numpy`` -- and on numpy for every dense
       decline/fallback (:class:`DenseUnsupported` included), because a per-tree
       iree compile (271-284 ms/tree) can never amortize.

    A ``FusedUnsupported`` from the engine at call time keeps the historical
    fallback chain (the fused route's ``numpy`` fallback is the specialized
    engine).  On a box with no device, and with an explicit backend, the
    historical rule is unchanged: dense iff the resolved backend is not ``numpy``
    *and* ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP`` *and*, on ``per_row=False``
    calls, the data-row count fits the route's MEMORY ENVELOPE (next paragraph);
    an explicit backend never reaches the fused route (the policy does not
    second-guess a named backend -- use ``EVOGP_EVAL_MODE=fused`` for that).  The
    wired call sites pass ``pop_size`` only, so the rows half of the rule is
    enforced inside :func:`evaluate_dense`, where the data-row count is known
    (:func:`sr_fitness_dense` goes through it too); ``per_row=True`` (one row per
    tree) and an explicit ``EVOGP_EVAL_MODE=dense`` are never gated, and a
    decline surfaces as the usual :class:`DenseUnsupported` fallback.
    Rationale: the dense path pays ONE compile for the whole population (3.1-3.4 s
    at ``L=32``, 6.1 s at the default cap ``L=64``) against 271-284 ms *per
    distinct structure* on iree, so the warm per-call win (0.5 ms at ``P=16``,
    ~7 ms at ``P=64``, ~34 ms at ``P=256``, ~141 ms at ``P=1000``) pays that
    compile back after ~6300/460/95/23 calls -- hence the ``pop_size`` gate at the
    population where a GP run is a few hundred generations in (lower
    ``EVOGP_DENSE_AUTO_MIN_POP`` to 16 for a very long run).

    The rows half is a **memory envelope, not a row ceiling**
    (:func:`auto_rows_decline`, :func:`auto_row_allowance`): what actually grows
    on this route -- and what can turn an ``auto`` decision into an OOM kill
    instead of a fallback -- is the ``pop_size * tree_len * rows`` product the
    graph indexes (``4`` bytes per element, the largest intermediate of
    :func:`dense_peak_bytes`), not the row count by itself.  ``auto`` therefore
    declines only when that product exceeds ``EVOGP_DENSE_AUTO_BUDGET_MB``
    (default :data:`DEFAULT_AUTO_BUDGET_MB` = 8192 MiB -- the same
    ``pop * tree_len * rows * 4`` rule the benchmark harness applies as
    ``CELL_TENSOR_CEILING_BYTES``, ``test/benchmark/evoxc/common.py``), and
    ``EVOGP_DENSE_AUTO_MAX_ROWS`` became a user PIN on the row allowance
    (unset/``0`` = unpinned: the envelope decides; a value can only TIGHTEN it,
    ``EVOGP_DENSE_AUTO_BUDGET_MB`` is the knob that widens it).
    Why the old fixed ceiling is gone: ``DEFAULT_AUTO_MAX_ROWS`` = 2048 was
    calibrated at ``P=16`` against the per-tree *device* backend (dense 1.18x
    slower at 10000 rows), a comparison the shipped policy never makes -- ``auto``
    admits no population below 64, and its fallback from a dense decline is
    specialized+``numpy``.  On the large-scale column that ceiling routed every
    cell with ``N >= 4096`` to a fallback dense beats by 1.4-8.7x per call (5.82
    against 51.90 ms at ``p256-L64-i32-N4096``, 78.12 against 107.59 at
    ``p256-L64-i32-N16384``, 39.31 against 195.51 at ``p1000-L64-i32-N4096``,
    117.19 against 452.28 at ``p1000-L64-i32-N16384``; ``test/benchmark/evoxc/
    LARGE_SCALE.md`` §7.1).  The part the fixed ceiling got right is kept where it
    was right: dense is never chosen on a non-device (``numpy``) backend at all,
    because there a compiled graph genuinely cannot amortize (it is 0.01-0.17x the
    specialized warm throughput) -- and that is the BACKEND term above, which needs
    no row count to be correct.

    The cap axis is the transposed twin of that envelope, and the same defect had
    the same shape: ``auto`` resolved dense on ``iree`` for a population whose
    realized trees exceed the trace cap, :func:`prepare_dense` then declined
    (``L = min(EVOGP_DENSE_MAX_TREE_LEN = 64, row_len = 128) = 64 < longest =
    123``), and the call site continued on ``route.fallback_backend`` =
    specialized+``numpy`` -- 857.22 ms where dense at the raised cap is 105.62 ms
    (``p2048-L128-i32-N1024-d7``; the opt-in batch route: 20.66 ms).  So under
    ``auto`` ONLY, and only when the cap was not user-pinned, the entry points now
    RAISE the cap for an over-cap population to
    ``min(EVOGP_DENSE_AUTO_MAX_TREE_LEN ceiling, EVOGP_DENSE_AUTO_BUDGET_MB
    envelope)`` ROUNDED to the next power of two up from the population's realized
    longest tree -- i.e. ``_next_power_of_two(max(base, longest))``, kept only
    while the allowance reaches that rounded value (:func:`auto_trace_cap`), from
    ONE memoized host read of ``batch_subtree_size[:, 0]``; nothing else --
    :func:`prepare_dense`, :func:`plan_dense`, :func:`resolve_route` -- changes.
    The rounding is what makes the raise friendly to a TRAINING LOOP: ``L`` is part
    of the dense graph key, so a cap that tracked ``longest`` exactly would compile
    a fresh graph every generation (``gp_step`` 10 014.83 ms at the drifting cap
    against 124.46 ms at a pinned ``L=128``), while a 65..128-node band shares one
    graph for the price of a few percent of trace work (103.34 ms at ``L=123``
    against ~105.6 ms at ``L=128``).  When the ceiling or the
    envelope refuses the raise the decline is DELIBERATE and falls back on the
    DEVICE backend instead of numpy (:class:`DenseUnsupported`'s
    ``device_fallback`` + :func:`decline_backend`), because per-tree numpy is the
    catastrophic answer there.  The residual limit is that specialized+``iree`` is
    per-structure-compile-bound, with unbounded artifact growth at large
    populations -- which is why the raise is bounded rather than unrestricted -- and
    that a generation whose longest tree crosses a power of two (128 -> 129) still
    recompiles once.
``dense``
    always attempt the dense path (still falling back on
    :class:`DenseUnsupported`); with no explicit backend it uses the device
    backend on a device box (a user forcing dense wants the fast dense route) and
    ``numpy`` elsewhere.
``specialized``
    never use dense; the specialized engine only, on the explicitly named (or
    engine-default) backend.
``batch``
    always attempt the OPT-IN batch route
    (:func:`evogp.tree.batch_engine.evaluate_batch` /
    :func:`evogp.tree.batch_engine.sr_fitness_batch`, ONE merged evoxc graph for
    the whole population), falling back on
    :class:`evogp.tree.batch_engine.BatchUnsupported`.  ``auto`` NEVER selects
    this engine, so it exists only behind an explicit
    ``EVOGP_EVAL_MODE=batch``; with no explicit backend it runs ``iree`` on a
    device box and ``numpy`` elsewhere, and its fallback is always
    specialized+``numpy`` (a per-tree iree compile can never amortize).
``fused``
    always attempt the OPT-IN fused route
    (:func:`evogp.tree.fused_engine.evaluate_fused` /
    :func:`evogp.tree.fused_engine.sr_fitness_fused`, ONE Triton launch for the
    whole population), falling back on
    :class:`evogp.tree.fused_engine.FusedUnsupported`.  It runs on the TORCH
    device rather than on an evoxc backend, so it uses the same ``iree`` backend
    SPELLING the other device routes use (the facade declines for any other name)
    and its fallback is ALWAYS specialized on ``numpy`` -- the explicit-backend
    arm included, even when ``EVOGP_EVOXC_BACKEND`` names ``iree``, because a
    per-tree iree compile can never amortize after a decline (the explicit
    spelling drives ``backend`` only, never ``fallback_backend``).  With no
    explicit backend it needs a device usable from torch and
    ``pop_size >= EVOGP_FUSED_MIN_POP`` (default 16 -- the kernel is flat in
    ``pop_size``, 0.088-0.103 ms from pop 1 to 128, so the threshold only keeps a
    one-shot ``Tree`` off a cold Triton JIT); below that threshold, or with no
    device, the route IS specialized+``numpy`` (the engine field stays
    ``"specialized"``).  ``auto`` also SELECTS this engine -- it is the default on
    a device box (:func:`auto_fused_decline`), on the same two gates plus the
    engine's memory envelope -- so an explicit ``EVOGP_EVAL_MODE=fused`` is only
    needed to force it on a box ``auto`` would route elsewhere.

Any other value raises ``ValueError`` naming the variable.  The dense-specific
knobs below stay as they are.

Scheme (see :func:`_dense_body` for the trace-time body)
-------------------------------------------------------
1. **Host preprocessing** (:func:`prepare_dense`) -- bulk-vectorized numpy over
   the ``(pop_size, max_tree_len)`` population tensors, no per-tree Python loop.
   It produces ``(L, P)``-shaped runtime *input* tensors: ``opcode`` (0 VAR,
   1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC/unknown-base, 5 inactive/padding),
   ``func_id`` (truncated toward zero, clamped to ``[0, 29]``), ``const_value``
   (float32), ``var_index`` (truncated, validated, clamped) and the three prefix
   child positions ``child1/2/3`` (clamped to ``[0, L)``).
   ``L = min(cap, max_tree_len)`` where ``cap`` is :data:`MAX_TREE_LEN_ENV`
   (default :data:`DEFAULT_MAX_TREE_LEN` = 64, so a ``max_tree_len=32`` population
   still traces ``L=32``); a population whose longest tree exceeds ``L`` -- or that
   fails *any* structural check -- raises :class:`DenseUnsupported` so the caller
   falls back to the specialized path (which owns the error contract; this module
   deliberately does NOT reproduce
   :func:`evoxc_engine.decode_population`'s error messages).  An over-cap tree is
   the ONE decline that warns (once per process, on a device backend -- see
   :func:`_warn_trace_cap`), because the specialized fallback there is the
   expensive one.  The ``auto`` ENTRY POINTS (:func:`evaluate_dense` /
   :func:`sr_fitness_dense`) may RAISE the cap for an over-cap population BEFORE
   :func:`prepare_dense` sees it (:func:`auto_trace_cap`, UP to a power of two so a
   band of realized lengths shares one compiled graph), so the preparation
   itself is unchanged and still declines above ``L``.
2. **One graph** (module-level ``@evoxc.defn _dense_step``) with a Python
   trace-time ``for i in range(L-1, -1, -1)`` unroll over node positions
   (reverse prefix order: children always have a larger index than their
   parent).  Per position it gathers the three child values (single-axis
   ``gather`` from a flat view of the value buffer -- the documented flatten
   trick), computes the whole function-candidate chain with the *same* private
   emitters the specialized engine uses (:func:`evoxc_engine._emit_unary` /
   :func:`_emit_binary` / :func:`_emit_ternary`, parity by construction) and
   selects the leaf / function value with broadcast ``select`` preds.
3. **No per-tree host calls, no per-tree readbacks**: the nodes are inputs, the
   graph keeps one row of the value buffer per position, and the caller reads
   back the assembled result ONCE per chunk (numpy) or once per chunk from the
   device (IREE).

Value buffer & write modes (:data:`UPDATE_MODE_ENV`)
---------------------------------------------------
The trace keeps already-computed values as a buffer of shape ``(L, P, N)``
(``N == 1`` for ``per_row=True``).  evoxc has no in-place mutation primitive that
is safe on every backend, so the row write is emulated:

``grow`` (default)
    The buffer starts as the row computed at ``i = L-1`` and is *prepended* to
    with ``concatenate([row, buf], axis=0)``; reads use the offset
    ``child - (i+1)``, clamped.  Row ``L-1`` is provably leaf-only for every
    validated tree (a function node there would need a child outside the row),
    so its three gathers are replaced by a constant.  Costs
    ``L*(L+1)/2 * P*N`` copied elements instead of ``L*L*P*N``, needs no initial
    buffer constant and uses only backend-agnostic ops.
``concat``
    A full ``(L, P, N)`` zero buffer (an embedded graph constant) written with
    ``concatenate([head, row, tail], axis=0)`` over static ``slice`` pieces --
    the iree-safe "select-mask" update without a boolean mask tensor.  Reads are
    plain ``child`` indices.
``scatter``
    A full zero buffer written with ``scatter(buf, i, row, axis=0)``.
    **numpy backend only by default**: on iree-cuda evoxc's ``scatter`` mutates its
    input tensor in place (a retained input is modified by later invokes), so
    iree refuses unless :data:`ALLOW_UNSAFE_SCATTER_ENV` is set
    (``dense_spike_check.py`` uses it for the optional "is an
    in-graph-constructed buffer safe?" experiment).

Chunking / peak memory
----------------------
The graph shaves ``P`` and ``N`` into fixed chunk sizes so one compile serves
the whole call (the last chunk is **padded** to the chunk size -- one shape per
compile).  The documented model (:func:`dense_peak_bytes`) is::

    peak_bytes(L, P, N, mode) = 4 * (buf_factor*L*P*N + (LIVE + 8)*P*N + 4*L*P)

with ``buf_factor = 2`` for ``grow`` and ``3`` otherwise, and ``LIVE = 40`` live
``(P, N)`` SSA candidates (the 28 function candidates + 3 gathered children +
leaf/misc).  ``N`` is shaved first, then ``P``, so that ``peak_bytes <=
budget`` (:data:`BUDGET_MB_ENV`, default 2048 MB).  This is a *design* target for
the peak, not a measured RSS.

Environment knobs
-----------------
=============================== =========== ==================================
``EVOGP_EVAL_MODE``             auto        ``dense`` / ``specialized`` / ``auto``
                                            (the master switch above)
``EVOGP_DENSE_AUTO_MIN_POP``    64          the ``auto`` policy's pop-size
                                            threshold (dense+iree from here up
                                            on a device box, specialized+numpy
                                            below it)
``EVOGP_DENSE_AUTO_MAX_ROWS``    unset/0     user PIN on the ``auto`` policy's row
                                            allowance on ``per_row=False``
                                            calls: it can only TIGHTEN the
                                            memory envelope below (an explicit
                                            row ceiling, kept for compatibility);
                                            ``per_row=True`` and an explicit
                                            ``dense`` are never gated
``EVOGP_DENSE_AUTO_BUDGET_MB``  8192        the memory envelope of that rule:
                                            ``auto`` declines when
                                            ``pop_size * tree_len * rows * 4``
                                            bytes exceed it (the harness's
                                            ``CELL_TENSOR_CEILING_BYTES`` rule); a
                                            decline -> ``DenseUnsupported``
                                            fallback; ``per_row=True`` and an
                                            explicit ``dense`` are never gated
``EVOGP_DENSE_AUTO_MAX_TREE_LEN``            256         ceiling (in nodes) of the
                                            ``auto`` automatic trace-cap
                                            raise: an over-cap population is
                                            traced at the population's realized
                                            longest tree ROUNDED UP to a power
                                            of two, kept only while
                                            ``min(ceiling, envelope)`` reaches
                                            that rounded value (a band of
                                            lengths then shares ONE compiled
                                            graph) instead of falling back;
                                            ``auto`` only, and a pinned
                                            ``EVOGP_DENSE_MAX_TREE_LEN`` /
                                            ``max_tree_len=`` wins; the raise is
                                            skipped entirely -- no host read at
                                            all -- for a pinned cap or a
                                            non-``auto`` mode
``EVOGP_DENSE_MAX_TREE_LEN``    64          max trace length ``L``; longer trees
                                            -> :class:`DenseUnsupported`
``EVOGP_DENSE_BUDGET_MB``       2048        chunking memory budget (MiB)
``EVOGP_DENSE_UPDATE_MODE``     grow        ``grow`` / ``concat`` / ``scatter``
``EVOGP_DENSE_ALLOW_UNSAFE_SCATTER``        unset  permits ``scatter`` on iree
``EVOGP_DENSE_HOST_MEMO``       16          entries in the plan/input front
                                            tables (``0`` disables them)
``EVOGP_EVOXC_ARTIFACT_DIR``    unset       opt-in on-disk executable artifact
                                            cache, shared with the specialized
                                            route (off when unset/empty)
=============================== =========== ==================================
Every knob is read per call (see the module's configuration section), so a
process can change strategy between calls.  Backend/device/opt-level knobs, the
shared compile cache (:data:`DENSE_CACHE` is the very same
:data:`evoxc_engine.CACHE`) and the ``EVOXC_DISABLE_LOCATIONS=1`` tracing speedup
are inherited unchanged from :mod:`evogp.tree.evoxc_engine`.
The disk layer (:mod:`evogp.tree.artifact_cache`) is the only thing here that
survives a process: one artifact per (shape, used-id set) for this route, loaded
instead of compiled by a later process; see :func:`build_dense` and
:func:`evoxc_engine.artifact_cache_stats`.

Parity notes / known numeric deviations
---------------------------------------
* Function and leaf values go through the identical emission path, so a dense
  value is bit-exact unless the *specialized* engine folded a constant subtree at
  decode time (``evoxc_engine._decode_row`` -> ``_fold_*``).  A folded chain
  rounds to float32 after every op, exactly like the emitted graph, so the only
  way to diverge is a host ``numpy`` float64 primitive (``pow``/``exp``/``sin``/
  ... in the fold path) disagreeing by one ulp with the backend's float64
  primitive before the float32 rounding -- ``dense_spike_check.py --full``
  searches for such cases and reports every non-bit-exact finite case.
* ``output_len > 1`` (multi-output trees, OUT_NODE tags, the out-node
  accumulation protocol) is NOT supported: :class:`DenseUnsupported`, so such
  populations go to the specialized engine (which implements the protocol).
  Note that the combined multi-output container
  (:class:`evogp.tree.CombinedForest`) is a list of *single-output* sub-forests
  and forwards through ``Forest.forward`` / ``Forest.batch_forward``, so its
  sub-forests take the same policy decision as any other population.
* ``per_row=True`` -> ``(pop_size, output_len)`` and ``per_row=False`` ->
  ``(pop_size, N, output_len)`` follow :func:`evoxc_engine.evaluate`'s contract
  (fresh, contiguous, ``requires_grad=False``, float32, on ``x``'s device unless
  ``out_device``); :func:`sr_fitness_dense` mirrors
  :func:`evoxc_engine.sr_fitness` (positive loss ``sum_N sum_out err / N``, MSE or
  MAE, NaN/inf propagate) with the very same torch reduction ops, so identical
  predictions give identical losses.

Shared cache, chunk policy, follow-ups
--------------------------------------
* The dense graphs live in the same ``evoxc_engine.CACHE`` as the specialized
  ones: one entry per shape signature, so ``evoxc_engine.cache_stats()`` reports
  the whole-population dense compile as ONE miss (against ``pop_size`` misses on
  the specialized path) and a later call of the same shape is a pure hit.
* **Chunking**: one compile must serve the whole call, so the graph's ``P``/``N``
  are fixed chunk sizes and the last chunk is padded up to the chunk size
  (:func:`_chunk_shape`): ``N`` is shaved first, then ``P``, until
  :func:`dense_peak_bytes` fits ``EVOGP_DENSE_BUDGET_MB``.  One call is therefore
  ``p_chunks * n_chunks`` executes of a single executable.
* Follow-ups only if the measurements justify them: cache device-resident prep
  tensors and ``x`` across calls (a generation re-evaluates a NEW population on
  the SAME ``x``); gate the per-position candidate chain by the position's
  possible opcodes; blocked writes (several rows per buffer update) to cut the
  ``L^2`` buffer traffic; an in-graph fitness reduction.

Measurements (round-1 battery, re-runnable with ``dense_spike_check.py``)
-------------------------------------------------------------------------
Run: repo root, ``PYTHONPATH=src EVOXC_DISABLE_LOCATIONS=1``, venv
``/mnt/local-ssd/bchuang/evox/.venv/bin/python``; GPU 3 via
``CUDA_VISIBLE_DEVICES=3`` + ``--gpu 0``.

1. **Correctness (numpy backend, CPU).** The full battery passes 34/34
   (``--backend numpy --quick``): random populations (4 descriptors/4 seeds ×
   ``per_row`` True/False × ``sr_fitness`` MSE/MAE), a 1200-tree function sweep
   over edge operands, unknown function ids / wrong-arity ids / unknown base
   types, padding-junk-tail invariance (same tree + different junk tails ->
   identical bits), fully constant trees, chunked execution (forced
   ``p_chunk=n_chunk=1``), the ``DenseUnsupported`` fallback signals, and
   one-compile-reuse (second call with a different population of the same shape
   does **0** builds).  Every finite value is bit-exact; the only bit
   differences are NaN payload bits (masks identical).
2. **Fold-vs-emit (numpy).** Zero non-bit-exact finite values across the fold
   search (4 const_prob levels × 2 seeds, 64 trees × 24 points) — expected, the
   emitters use the same f32-hardware / f64-transcendental discipline as
   ``_fold_*``.
3. **Fold-vs-emit (iree-cuda) — 1 ulp, and it is the ENGINE that moves.** With
   ``--backend iree`` the fold search reports ~24 non-bit-exact finite values per
   population at ``max_rel`` 6e-8 … 3.3e-7 (1 ulp at those magnitudes) while the
   dense result matches the engine's *emitted* path: the specialized engine folds
   constant subtrees host-side with **numpy f64** transcendentals, the dense
   graph evaluates them with **iree f64** transcendentals, and iree's f64 libm is
   not bit-identical to numpy's. Same class of ~1 ulp difference for
   ``const_out`` trees. All within the 1e-4 tolerance; report it, do not chase it.
4. **Known pre-existing ``evoxc_engine`` issue, iree + ``per_row=True``** (owned by
   the specialized engine, tracked/fixed separately; the dense path does not
   share that code).  ``evoxc_engine._run_tree`` (``evoxc_engine.py:1237-1242``)
   memoizes the host->device input as ``prepared[0]`` and reuses it for every
   tree of one ``evaluate`` call. That is right for ``per_row=False`` (same
   ``xin``) and WRONG for ``per_row=True`` (``xin = xt[index:index+1]``), where
   every tree silently runs on ``x``'s ROW 0. Minimal repro (2 ×
   ``SUB(var0,var1)``, ``x=[[10,1],[20,2]]``, expected ``[9,18]``): engine numpy
   ``[9,18]``, engine iree ``[9,9]``, ``evaluate_dense`` iree ``[9,18]``.
   Reachable through ``Forest.forward`` and therefore through ``BraxProblem`` /
   ``MujocoProblem`` rollouts -- the exact users of iree.  numpy (the default
   backend) is unaffected.
5. **iree runtime robustness.** A long battery (≈25 compiled executables in one
   process) segfaulted the interpreter mid-run, after emitting nanobind
   "leaked function/HalBuffer" warnings — the evoxc/iree binding leaks device
   buffers per live executable. Keep iree sessions short and split batteries
   into stages; ``cache_stats()``/``clear_cache()`` between stages.
6. **Economics (numpy CPU interpreter, warm medians, P/N from the script).**
   Dense is **much slower** than the specialized path on numpy: 0.01–0.17×
   (``P=64,N=64``: 18.8 ms specialized vs 209.7 ms dense; ``P=64,N=1024``:
   19.9 ms vs 3137.6 ms), because the graph carries ≈264 ops per traced
   position (≈264·L ops: L=8 → 2110, L=16 → 4478, L=32 → 9214) and evoxc's numpy
   interpreter pays per-op overhead on ``(L,P,N)`` tensors. Cold build is also
   worse (dense 5.1 s vs 196 ms for P=64) and both grow with L (numpy L=32:
   trace 204 ms, lower 468 ms, compile 1893 ms, load 397 ms).
   On iree the trade is the opposite shape: dense pays ONE compile per shape
   signature (7.2 s for L_trace=3/P=4/N=6 with ≈1.8k ops, i.e. iree compile
   cost scales with the op count) and then ~88 ms per call for P=4/N=6
   (dominated by the 8 host->device uploads per chunk, not the math), while the
   specialized path pays ~0.89 s of compile PER TREE (3.56 s cold for P=4) and
   ~1.9 ms of run per tree (7.6 ms for P=4). So dense-on-iree wins once
   ``P × 0.89 s`` exceeds one dense compile, i.e. from roughly **P ≳ 10–20 at
   L_trace ≈ 3**, and the crossover moves out linearly with L through the
   dense compile cost. On numpy the same comparison is hopeless at every P.
7. **Consequence for the integration policy** (this is what ``auto`` encodes).
   The dense path is a *shape-signature-specialized, one-compile-per-population*
   strategy whose realistic home is iree (or a future batched-kernel backend);
   its worst enemy is the per-position candidate chain.  So ``auto`` uses it on
   iree only when the population is dense-supported,
   ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP`` (64 since the pruned chain made the
   per-call win large; the pre-prune crossover was ~25-35 trees for the compile
   term) and, on ``per_row=False`` calls, while the data-row count fits the route's
   memory envelope ``pop_size * tree_len * rows * 4`` bytes <=
   ``EVOGP_DENSE_AUTO_BUDGET_MB`` (8192 MiB; an explicit
   ``EVOGP_DENSE_AUTO_MAX_ROWS`` can tighten it further; enforced inside
   :func:`evaluate_dense`, where the row count is known -- ``per_row=True`` and an
   explicit ``dense`` are never gated; a decline is the usual
   :class:`DenseUnsupported` fallback), and it never selects the numpy backend for
   the dense route on its own.
   The row ceiling itself is gone: the measured crossovers that motivated a fixed
   one (~3.4-6.8k rows at ``P=16``, dense 1.18x slower at ``P=16/N=10000``) were
   taken against the per-tree *iree* engine at a population ``auto`` no longer
   admits, and against every fallback ``auto`` can actually reach -- specialized on
   numpy -- dense is 1.4-8.7x FASTER at ``N`` = 4096-16384 and pop 256-1000
   (``test/benchmark/evoxc/LARGE_SCALE.md`` §7.1), so a row ceiling there routes
   the call the wrong way.  What is left to bound is the product the graph sizes
   its largest intermediate by, i.e. a memory envelope, not a row count.
   The cheapest large win identified: pass the population's
   *used function id set* (or its descriptor's ``using_funcs``) as a static cfg
   entry and emit only those candidates — roughly a 3× op-count cut, paid for by
   recompiles only when the used-id set changes (it saturates quickly in a GP
   run).  That moved the iree crossover down and is the ``EVOGP_DENSE_PRUNE``
   default; the compile numbers quoted in this module are the full-chain ones
   unless stated otherwise.

Benchmark-support runner (not part of the public route)
-------------------------------------------------------
:func:`build_dense_runner` / :class:`DenseRunner` expose the dense evaluator as
a build-once / run-many object for benchmarking harnesses: the build phase plans
+ compiles (or hits) the ONE shape-signature graph and stages every payload of
the (single) chunk through the readback-free seam
(:func:`_iter_plan_chunks` with an injected memo), then ``.run()`` executes only
the forward pass -- one ``evoxc.run`` over the staged payloads, returning the raw
backend result, with no readback, no host assembly, no re-plan and no key
derivation.  It changes nothing about ``evaluate_dense`` / ``sr_fitness_dense``,
is called by nothing in the library and is documented -- with the specialized
twin, ``evoxc_engine.build_specialized_runner`` -- in ``RUNNER_HOOKS.md``.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc

from .utils import NType
from . import artifact_cache, dense_memo
from .multi_output import check_multi_output
from .evoxc_engine import (  # private but deliberately reused: parity by construction
    BACKEND_ENV,
    CACHE,
    IREE_DEVICE_ENV,
    IREE_OPT_LEVEL_ENV,
    _CPU,
    _as_numpy,
    _emit_binary,
    _emit_ternary,
    _emit_unary,
    _env_int,
    _freeze_options,
    _operand,
    _population_inputs,
    _quiet,
    _register_cache_reset,
    _resolve_backend,
    _trace_const,
)

__all__ = [
    "ALLOW_UNSAFE_SCATTER_ENV",
    "AUTO_BUDGET_MB_ENV",
    "AUTO_MAX_ROWS_ENV",
    "AUTO_MAX_TREE_LEN_ENV",
    "AUTO_MIN_POP_ENV",
    "BUDGET_MB_ENV",
    "DEFAULT_AUTO_BUDGET_MB",
    "DEFAULT_AUTO_MAX_ROWS",
    "DEFAULT_AUTO_MAX_TREE_LEN",
    "DEFAULT_AUTO_MIN_POP",
    "DEFAULT_BUDGET_MB",
    "DEFAULT_EVAL_MODE",
    "DEFAULT_MAX_TREE_LEN",
    "DEFAULT_HOST_MEMO",
    "DEFAULT_PRUNE",
    "DENSE_ARTIFACT_NAMES",
    "DENSE_CACHE",
    "DENSE_EXECUTABLE_MEMO",
    "DENSE_INPUT_MEMO",
    "DENSE_PLAN_MEMO",
    "DenseEntry",
    "DensePlan",
    "DensePrep",
    "DenseRunner",
    "DenseUnsupported",
    "EVAL_MODES",
    "EVAL_MODE_ENV",
    "ENGINES",
    "FULL_FUNC_IDS",
    "FUSED_MIN_POP_ENV",
    "HOST_MEMO_ENV",
    "MAX_TREE_LEN_ENV",
    "PRUNE_ENV",
    "OP_BFUNC",
    "OP_CONST",
    "OP_INACTIVE",
    "OP_TFUNC",
    "OP_UFUNC",
    "OP_VAR",
    "Route",
    "UPDATE_MODES",
    "UPDATE_MODE_ENV",
    "auto_cap_decline",
    "auto_row_allowance",
    "auto_rows_decline",
    "auto_trace_cap",
    "auto_tree_len_allowance",
    "build_dense",
    "build_dense_runner",
    "clear_route_memo",
    "decline_backend",
    "dense_cache_stats",
    "dense_peak_bytes",
    "describe_plan",
    "device_available",
    "evaluate_dense",
    "plan_dense",
    "prepare_dense",
    "resolve_auto_budget_mb",
    "resolve_auto_max_rows",
    "resolve_auto_max_tree_len",
    "resolve_auto_min_pop",
    "resolve_backend_name",
    "resolve_eval_mode",
    "resolve_fused_min_pop",
    "resolve_prune",
    "resolve_route",
    "route_cache_stats",
    "should_use_dense",
    "sr_fitness_dense",
    "trace_cap_pinned",
]

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: Master switch (``EVOGP_EVAL_MODE``): ``dense`` / ``specialized`` / ``batch`` /
#: ``fused`` / ``auto``.  The ONE policy knob; :func:`resolve_eval_mode` /
#: :func:`resolve_route` read it per call, exactly like
#: ``evoxc_engine._resolve_backend`` reads the backend variable.
EVAL_MODE_ENV = "EVOGP_EVAL_MODE"
#: Population-size threshold of the ``auto`` policy
#: (``EVOGP_DENSE_AUTO_MIN_POP``): dense+iree from here up on a device box,
#: specialized+numpy below it -- see :func:`resolve_route`.
AUTO_MIN_POP_ENV = "EVOGP_DENSE_AUTO_MIN_POP"
#: User PIN on the ``auto`` policy's data-row allowance
#: (``EVOGP_DENSE_AUTO_MAX_ROWS``) for ``per_row=False`` calls.  It is an explicit
#: row ceiling the user can still set, and it can only TIGHTEN the memory envelope
#: of :data:`AUTO_BUDGET_MB_ENV` (:func:`auto_row_allowance`): unset/empty/``0``
#: means "unpinned, the envelope decides", ``N`` means "at most ``N`` rows, or the
#: envelope's allowance if that is smaller".  See :func:`auto_rows_decline`.
AUTO_MAX_ROWS_ENV = "EVOGP_DENSE_AUTO_MAX_ROWS"
#: Memory envelope of the ``auto`` policy's rows rule
#: (``EVOGP_DENSE_AUTO_BUDGET_MB``, MiB): dense's largest intermediate is
#: ``pop_size * tree_len * rows * 4`` bytes, so ``auto`` declines a call whose
#: product exceeds it -- the same rule the benchmark harness enforces as
#: ``CELL_TENSOR_CEILING_BYTES`` (``test/benchmark/evoxc/common.py``).  This is the
#: knob that WIDENS the allowance (the row pin above only tightens it) -- see
#: :func:`auto_rows_decline` / :func:`auto_row_allowance`.
AUTO_BUDGET_MB_ENV = "EVOGP_DENSE_AUTO_BUDGET_MB"
#: Ceiling of the ``auto`` policy's automatic trace-cap raise
#: (``EVOGP_DENSE_AUTO_MAX_TREE_LEN``, in NODES): the cap-axis twin of
#: :data:`AUTO_BUDGET_MB_ENV`.  Under ``auto`` (and only there, and only when the
#: cap was not pinned by ``EVOGP_DENSE_MAX_TREE_LEN``/``max_tree_len=``) a
#: population whose realized longest tree exceeds the trace cap is evaluated at
#: ``_next_power_of_two(max(base, longest))`` -- the realized length rounded UP, so
#: one compiled graph spans a whole band of lengths -- instead of falling back to
#: the specialized engine, and only while ``min(ceiling, envelope)`` reaches that
#: rounded value -- dense's trace length, graph size and compile all grow
#: linearly in ``L`` (~13.9 s at ``L=32``, ~31.6 s at ``L=64``, ~69.0 s at
#: ``L=128`` per shape signature), so the raise needs a bound.  See
#: :func:`auto_tree_len_allowance` / :func:`auto_trace_cap` / :func:`auto_cap_decline`.
AUTO_MAX_TREE_LEN_ENV = "EVOGP_DENSE_AUTO_MAX_TREE_LEN"
#: Pop-size threshold of the ``fused`` policy (``EVOGP_FUSED_MIN_POP``): the Triton
#: route is attempted from here up on a device box, and below it the route IS
#: specialized+numpy.  The kernel is FLAT in ``pop_size`` (0.088-0.103 ms from pop 1
#: to 128 at ``L=64``/``i32``/``N=1024``) while ``specialized`` scales with the tree
#: count (5.3x / 26.5x / 244.7x dearer at pop 1 / 16 / 128), so the number is set by
#: what a one-shot call may pay rather than by the kernel's own crossover: 16 keeps a
#: single ``Tree.forward`` on the specialized route instead of a cold Triton JIT, and
#: a 1-8-tree population on a tiny row plane is the one unmeasured class where a
#: per-tree numpy run could undercut the kernel's ~0.09 ms launch floor.
#: The variable has EXACTLY ONE parser,
#: :func:`evogp.tree.fused_engine.resolve_min_pop`; :func:`resolve_fused_min_pop`
#: delegates to it -- see that function.
FUSED_MIN_POP_ENV = "EVOGP_FUSED_MIN_POP"
#: The fused route's memory-envelope knobs, re-declared for :func:`_route_key` ONLY
#: (the route policy reads them through ``fused_engine.envelope_decline`` when a
#: caller passes ``rows``/``tree_len``, so a memoized decision must be invalidated by
#: a flip).  They are OWNED -- the parsing, the defaults and the error messages -- by
#: :mod:`evogp.tree.fused_engine` (:data:`~evogp.tree.fused_engine.MAX_SCRATCH_MB_ENV`,
#: :data:`~evogp.tree.fused_engine.BLOCK_ROWS_ENV`,
#: :data:`~evogp.tree.fused_engine.NUM_WARPS_ENV` there, the three module attributes
#: :func:`~evogp.tree.fused_engine.resolve_max_scratch_mb` and its siblings read); a
#: test pins the two spellings together so they cannot drift.
FUSED_MAX_SCRATCH_MB_ENV = "EVOGP_FUSED_MAX_SCRATCH_MB"
FUSED_BLOCK_ROWS_ENV = "EVOGP_FUSED_BLOCK_ROWS"
FUSED_NUM_WARPS_ENV = "EVOGP_FUSED_NUM_WARPS"
#: Max trace length ``L`` (``EVOGP_DENSE_MAX_TREE_LEN``): populations whose
#: longest tree exceeds it fall back to the specialized engine.  This is the
#: knob that bounds trace time and graph size (the candidate chain is emitted
#: ``L`` times) -- see the ``L`` scaling curve in the module docstring and
#: :data:`DEFAULT_MAX_TREE_LEN` for the measured default.
MAX_TREE_LEN_ENV = "EVOGP_DENSE_MAX_TREE_LEN"
#: Chunking memory budget in MiB (``EVOGP_DENSE_BUDGET_MB``).
BUDGET_MB_ENV = "EVOGP_DENSE_BUDGET_MB"
#: Buffer write strategy (``EVOGP_DENSE_UPDATE_MODE``): see the module docstring.
UPDATE_MODE_ENV = "EVOGP_DENSE_UPDATE_MODE"
#: Explicit opt-in for ``scatter`` on IREE, where evoxc's scatter mutates its input.
ALLOW_UNSAFE_SCATTER_ENV = "EVOGP_DENSE_ALLOW_UNSAFE_SCATTER"

#: Canonical ``EVOGP_EVAL_MODE`` spellings; ``auto`` stays LAST.  ``batch`` is the
#: opt-in third engine (:mod:`evogp.tree.batch_engine`), ``fused`` the opt-in
#: fourth (:mod:`evogp.tree.fused_engine`), and neither is ever produced by the
#: ``auto`` policy -- see :func:`resolve_route`.
EVAL_MODES = ("dense", "specialized", "batch", "fused", "auto")
DEFAULT_EVAL_MODE = "auto"
#: The four evaluation engines a :class:`Route` can name: ``specialized``
#: (one graph per tree structure), ``dense`` (one all-ops graph per shape),
#: ``batch`` (one merged graph for the whole population, opt-in only) and
#: ``fused`` (one Triton launch per population, opt-in only).
ENGINES = ("specialized", "dense", "batch", "fused")
#: Accepted ALIASES of the canonical :data:`EVAL_MODES`, resolved by
#: :func:`resolve_eval_mode` after case folding/trimming and before validation
#: (so ``EVOGP_EVAL_MODE=dynamic`` and ``eval_mode="dynamic"`` behave exactly
#: like ``dense``, the ``auto`` rows rule included).  ``dynamic`` names what the
#: dense route actually *is*: one compiled "dynamic all-ops" graph that keeps the
#: tree structure as runtime DATA (every candidate op is emitted once and the
#: data-driven ``select`` chain picks the node's operation at run time), against
#: the specialized engine's one-structure-per-graph strategy.  It is a
#: first-class spelling, not a separate mode, so :data:`EVAL_MODES` stays
#: canonical and :func:`should_use_dense` keeps branching on ``"dense"``.
EVAL_MODE_ALIASES = {"dynamic": "dense"}

#: ``DensePlan.variant`` values: the value graph (:func:`_dense_step`, the
#: default) and the SR-loss graph (:func:`_dense_loss_step`, which folds the
#: fitness reduction in so only ``(pop_size,)``-scale losses leave the engine).
VALUE_VARIANT = "value"
LOSS_VARIANT = "loss"
PLAN_VARIANTS = (VALUE_VARIANT, LOSS_VARIANT)

#: Canonical graph name of each variant in an on-disk artifact key
#: (``evogp.tree.artifact_cache``): the process-stable stand-in for the ``id()`` of
#: :func:`_dense_step` / :func:`_dense_loss_step`.  The plan's whole spec tuple
#: (shapes, ``x``, the loss's ``labels``/``keep`` and the static ``cfg``) is part of
#: the key, so one variant keeps one artifact per (shape, used-id set).
DENSE_ARTIFACT_NAMES = {VALUE_VARIANT: "dense:value", LOSS_VARIANT: "dense:loss"}

#: Pop-size threshold of the ``auto`` policy, measured on the current pruned
#: dense chain (one dense compile at ``L=32`` on iree is 3.1-3.4 s / 6.1 s at
#: ``L=64``): the warm per-call win of dense+iree over specialized+numpy is
#: 0.5 ms at ``P=16``, ~7 ms at ``P=64``, ~34 ms at ``P=256`` and ~141 ms at
#: ``P=1000``, i.e. the compile is paid back after ~6300 / ~460 / ~95 / ~23
#: calls.  64 is the population where a GP run is a few hundred generations in;
#: lower it to 16 (``EVOGP_DENSE_AUTO_MIN_POP=16``) for a very long run.
DEFAULT_AUTO_MIN_POP = 64
#: Default of the ``auto`` rows PIN (:data:`AUTO_MAX_ROWS_ENV`): ``0`` = unpinned,
#: i.e. the memory envelope (:data:`DEFAULT_AUTO_BUDGET_MB`) decides the allowance.
#: The historical default was a fixed 2048-row ceiling, calibrated on iree (RTX
#: A6000, ``L=32``) at ``P=16`` against the per-tree *device* engine (dense 1.49x /
#: 1.30x faster at 64 / 1024 rows, 1.18x *slower* at 10000) -- a comparison the
#: shipped policy never makes, since ``auto`` admits no population below
#: ``DEFAULT_AUTO_MIN_POP`` and its fallback from a dense decline is
#: specialized+``numpy``, which dense beats by 1.4-8.7x at ``N`` = 4096-16384
#: (``test/benchmark/evoxc/LARGE_SCALE.md`` §7.1) and by 1.6-8.7x per generation
#: there.  Hence the pin is off by default and the envelope does the work.
DEFAULT_AUTO_MAX_ROWS = 0
#: Default of the ``auto`` rows rule's memory envelope (:data:`AUTO_BUDGET_MB_ENV`),
#: in MiB: 8192, the same ``pop * tree_len * rows * 4`` ceiling the benchmark
#: harness applies as ``CELL_TENSOR_CEILING_BYTES`` (``test/benchmark/evoxc/
#: common.py``).  At that envelope the worst large-scale cell
#: (``p1000-L64-i32-N16384``, 4.19 GB) keeps 2x of headroom, and dense's own
#: execution peak is bounded separately by :data:`DEFAULT_BUDGET_MB` chunking, so
#: this knob is the "when does the whole-population shape stop being schedulable"
#: line, not an OOM tripwire.
DEFAULT_AUTO_BUDGET_MB = 8192
#: Default ceiling (in nodes) of the ``auto`` trace-cap raise
#: (:data:`AUTO_MAX_TREE_LEN_ENV`): 256, a deliberate bound on trace length, graph
#: size and compile rather than a measured optimum.  The graph's cost and its
#: compile both grow linearly in the trace length (~13.9 s at ``L=32``, ~31.6 s at
#: ``L=64``, ~69.0 s at ``L=128`` per shape signature on iree), so the raise that
#: rescues an over-cap population is capped well below the shipped descriptors'
#: declared ``max_tree_len`` (128/256/512) while still covering their REALIZED
#: trees (123-201 nodes on the large-scale column).  A user who knows better pins
#: ``EVOGP_DENSE_MAX_TREE_LEN`` instead, which this raise never touches.
DEFAULT_AUTO_MAX_TREE_LEN = 256

#: Trace cap default: 64, i.e. the shipped descriptors' practical maximum
#: (``example/`` builds ``max_tree_len`` 128/256/512, whose REALIZED trees are far
#: shorter).  Measured on iree (the item-5 measurement round): one compile costs
#: ~13.9 s (L=32) / ~31.6 s (L=64) / ~69.0 s (L=128) per shape signature, so a
#: 33-64-node population at P=64 (1 build, ~28.9 s cold, ~11.6 ms warm) beats the
#: specialized fallback there (64 builds, ~48.8 s cold, ~110.3 ms warm;
#: break-even ~45-60 distinct structures), while at P=16 it trades 4.1x cold for
#: 4.9x warm.  L=128 is rejected: it would tax every ``row_len >= 128`` population
#: -- the shipped examples are 128/256/512 -- with a ~69 s compile.  ``L`` is
#: ``min(cap, row_len)``, so short rows are unaffected by the default.
DEFAULT_MAX_TREE_LEN = 64
DEFAULT_BUDGET_MB = 2048
UPDATE_MODES = ("grow", "concat", "scatter")
DEFAULT_UPDATE_MODE = "grow"
#: Candidate-chain pruning (``EVOGP_DENSE_PRUNE``): emit the per-position
#: candidate ops only for the function ids the POPULATION actually uses
#: (:attr:`DensePrep.used_func_ids`) instead of all 28, which is where the bulk of
#: the ``~264*L`` emitted ops goes.  On by default; ``=0`` restores the full chain
#: byte-for-byte (the in-process "before" of an A/B).
PRUNE_ENV = "EVOGP_DENSE_PRUNE"
DEFAULT_PRUNE = True

#: The dense graphs live in the *same* compile cache as the specialized ones.
DENSE_CACHE = CACHE

#: Entries of the population/input front tables (``EVOGP_DENSE_HOST_MEMO``).
#: Deliberately small: a plan entry holds the decoded host columns of a whole
#: population (``8 * L * P`` float32/int32 bytes) and its staged device payloads,
#: and a training loop presents a NEW population every generation -- only the
#: repeated evaluation of an unchanged population (elites, benchmarks, a
#: ``forward`` followed by ``sr_fitness``) benefits, so a large table would only
#: pin memory.  ``0`` disables both tables (every call then does the full host
#: work, i.e. the pre-memo behaviour).
HOST_MEMO_ENV = "EVOGP_DENSE_HOST_MEMO"
DEFAULT_HOST_MEMO = 16


def _resolve_host_memo_entries(value: Optional[int] = None) -> int:
    """Resolve ``EVOGP_DENSE_HOST_MEMO`` (explicit argument wins)."""
    if value is None:
        raw = os.environ.get(HOST_MEMO_ENV, "").strip()
        if not raw:
            return DEFAULT_HOST_MEMO
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{HOST_MEMO_ENV}={raw!r} is not an integer"
            ) from exc
    entries = int(value)
    if entries < 0:
        raise ValueError(f"{HOST_MEMO_ENV}={entries} must be >= 0")
    return entries


#: Dense executables: everything ``evoxc.build_cached`` keys on, sized like the
#: compile cache itself (an entry is the executable plus a small key tuple).
DENSE_EXECUTABLE_MEMO = dense_memo.register(
    dense_memo.DenseMemo("dense_exec", DENSE_CACHE.maxsize)
)
#: :class:`DensePlan` (+ executable + staged payloads) of an unchanged population.
DENSE_PLAN_MEMO = dense_memo.register(
    dense_memo.DenseMemo("dense_plan", _resolve_host_memo_entries())
)
#: Host copy of an unchanged ``x`` / ``labels`` tensor (skips the per-call D2H).
DENSE_INPUT_MEMO = dense_memo.register(
    dense_memo.DenseMemo("dense_input", _resolve_host_memo_entries())
)
#: :func:`evoxc_engine.clear_cache` owns the invalidation of all three.
_register_cache_reset(dense_memo.clear)


def dense_cache_stats() -> Dict[str, int]:
    """Counters of the dense route's three front tables (see :mod:`dense_memo`).

    Deliberately NOT folded into :func:`evoxc_engine.cache_stats`, whose eight
    keys describe the compile cache and the specialized executable memo and are
    pinned by the test suite: ``dense_exec_*`` are the dense executables (a memo
    hit never reaches ``build_cached``, so ``CACHE.hits`` no longer grows with a
    repeated ``build_dense``), ``dense_plan_*`` the per-population plan/payload
    table and ``dense_input_*`` the resident ``x``/``labels`` host copies.
    """
    return dense_memo.stats()


#: ``opcode`` values (mirroring ``NType`` base values, plus 5 for padding).
OP_VAR = int(NType.VAR)  # 0
OP_CONST = int(NType.CONST)  # 1
OP_UFUNC = int(NType.UFUNC)  # 2 (arity 1)
OP_BFUNC = int(NType.BFUNC)  # 3 (arity 2)
OP_TFUNC = int(NType.TFUNC)  # 4 (arity 3 -- also every unknown base type)
OP_INACTIVE = 5  # padding / tree-length tail (all masks false -> value 0.0)

#: Highest function id the candidate chain dispatches on (``Func.LOOSE_SQRT``).
_MAX_FUNC_ID = 28
#: Highest binary (arity-2) function id (``Func.GE``); ids above it are unary.
_BFUNC_MAX = 13
#: Clamp for the truncated function id: ids ``>= 29`` are unmatched everywhere in
#: the engine and therefore yield 0.0; clamping keeps the int32 chain in range.
_FUNC_CLAMP = 29
#: The full candidate-id range: what the chain emits with
#: :data:`PRUNE_ENV` off, and the identity a pruned set is a subset of.  Id 0
#: (``Func.IF``) is the separately emitted ternary, and ids outside ``1..28``
#: match nothing anywhere (both are therefore no candidate emission at all).
FULL_FUNC_IDS = tuple(range(1, _MAX_FUNC_ID + 1))
#: Live ``(P, N)`` SSA candidates counted by :func:`dense_peak_bytes`.
_LIVE_CANDIDATES = 40

class DenseUnsupported(RuntimeError):
    """The dense path cannot (safely) evaluate this population.

    Callers must fall back to the specialized engine
    (:func:`evoxc_engine.decode_population` + :func:`evoxc_engine.evaluate`), which
    owns the error contract for malformed trees.  ``reason`` carries the
    human-readable cause; this module deliberately does not reproduce the
    specialized decoder's ``ValueError`` messages.

    ``device_fallback`` marks the ONE class of decline where the fallback should
    stay on the DEVICE backend instead of ``route.fallback_backend`` (numpy): the
    ``auto`` trace-cap/envelope decline, where per-tree numpy is the catastrophic
    answer (857.22 ms against 105.62 ms measured at 123-node trees, pop 2 048,
    ``N=1024``).  It is opt-in and defaults to ``False``, so a positional
    ``DenseUnsupported(reason)`` caller behaves exactly as before; the call sites
    go through :func:`decline_backend`.
    """

    def __init__(self, reason: str, *, device_fallback: bool = False):
        super().__init__(f"dense evaluation unsupported: {reason}")
        self.reason = reason
        self.device_fallback = bool(device_fallback)


def _resolve_max_tree_len(value: Optional[int]) -> int:
    cap = (
        _env_int(MAX_TREE_LEN_ENV, DEFAULT_MAX_TREE_LEN)
        if value is None
        else int(value)
    )
    if cap <= 0:
        raise ValueError(f"max_tree_len={cap} must be positive")
    return cap


def _resolve_budget_mb(value: Optional[float]) -> float:
    if value is None:
        raw = os.environ.get(BUDGET_MB_ENV, "").strip()
        budget = float(raw) if raw else float(DEFAULT_BUDGET_MB)
    else:
        budget = float(value)
    if budget <= 0:
        raise ValueError(f"budget_mb={budget} must be positive")
    return budget


def resolve_eval_mode(value: Optional[str] = None) -> str:
    """Resolve the ``EVOGP_EVAL_MODE`` master switch (explicit argument wins).

    Read per call -- like every knob here and like ``evoxc_engine._resolve_backend``
    reads ``EVOGP_EVOXC_BACKEND`` -- so a process can change strategy between calls
    (e.g. force ``specialized`` to produce a reference result).  Case-insensitive;
    an empty value means the default ``"auto"``; the accepted aliases
    (:data:`EVAL_MODE_ALIASES`, currently ``dynamic`` -> ``dense``) are folded in
    before validation; anything else raises ``ValueError`` naming the variable.
    """
    raw = value if value is not None else os.environ.get(EVAL_MODE_ENV, "")
    mode = str(raw).strip().lower()
    if not mode:
        mode = DEFAULT_EVAL_MODE
    mode = EVAL_MODE_ALIASES.get(mode, mode)
    if mode not in EVAL_MODES:
        aliases = ", ".join(
            f"{alias!r} = {target!r}" for alias, target in EVAL_MODE_ALIASES.items()
        )
        raise ValueError(
            f"unsupported eval mode {mode!r}: evogp.tree.dense_engine supports "
            f"{list(EVAL_MODES)} (plus the aliases {aliases}) (set "
            f"{EVAL_MODE_ENV!r} or pass eval_mode=); 'dense' (alias 'dynamic') = "
            "always try the dense evaluator and fall back on DenseUnsupported, "
            "'batch' = always try the opt-in merged-graph (batch) evaluator and "
            "fall back on BatchUnsupported, "
            "'fused' = always try the opt-in fused (Triton) evaluator and fall "
            "back on FusedUnsupported, "
            "'specialized' = never use it, 'auto' = the measured policy"
        )
    return mode


def resolve_backend_name(
    backend: Optional[str] = None, iree_device_index: Optional[int] = None
) -> str:
    """The backend name a call will run on, resolved the engine's way.

    Single-sources the ``explicit argument > EVOGP_EVOXC_BACKEND > default`` rule
    (and the backend-name validation) by delegating to
    :func:`evoxc_engine._resolve_backend`; no graph is built, so this is the cheap
    way to ask "numpy or not?".
    """
    return _resolve_backend(backend, iree_device_index)[0]


def resolve_auto_min_pop(value: Optional[int] = None) -> int:
    """Resolve the ``auto`` policy's pop-size threshold (``EVOGP_DENSE_AUTO_MIN_POP``)."""
    minimum = (
        _env_int(AUTO_MIN_POP_ENV, DEFAULT_AUTO_MIN_POP)
        if value is None
        else int(value)
    )
    if minimum < 1:
        raise ValueError(
            f"{AUTO_MIN_POP_ENV}={minimum} must be >= 1 (it is a pop_size threshold)"
        )
    return minimum
def resolve_fused_min_pop(value: Optional[int] = None) -> int:
    """Resolve the ``fused`` policy's pop-size threshold (``EVOGP_FUSED_MIN_POP``).

    The threshold has EXACTLY ONE parser -- :func:`evogp.tree.fused_engine.resolve_min_pop`,
    which reads the same variable and owns its validation, its default (16, the
    measured flat-kernel threshold) and its error message --
    so this reader only delegates to it.  The import is LOCAL on purpose: the
    route policy is both the first thing every call site touches and the module
    ``fused_engine`` must not depend on, so a module-level import would reverse
    the one-way module DAG (and pull the Triton route into every
    ``import evogp.tree``).
    """
    from . import fused_engine

    return int(fused_engine.resolve_min_pop(value))
def resolve_auto_max_rows(value: Optional[int] = None) -> int:
    """Resolve the ``auto`` rows PIN (``EVOGP_DENSE_AUTO_MAX_ROWS``), ``0`` = unpinned.

    A positive value is a hard row ceiling the user can still set; ``0`` (the
    default, :data:`DEFAULT_AUTO_MAX_ROWS`) means "let the memory envelope
    (:func:`resolve_auto_budget_mb`) decide".  The pin can only TIGHTEN the
    envelope's allowance -- see :func:`auto_row_allowance` -- because widening it
    is what ``EVOGP_DENSE_AUTO_BUDGET_MB`` is for.
    """
    maximum = (
        _env_int(AUTO_MAX_ROWS_ENV, DEFAULT_AUTO_MAX_ROWS)
        if value is None
        else int(value)
    )
    if maximum < 0:
        raise ValueError(
            f"{AUTO_MAX_ROWS_ENV}={maximum} must be >= 0 (0 = unpinned, else it is "
            "a data-row ceiling)"
        )
    return maximum


def resolve_auto_budget_mb(value: Optional[int] = None) -> int:
    """Resolve the ``auto`` rows rule's memory envelope (``EVOGP_DENSE_AUTO_BUDGET_MB``)."""
    budget = (
        _env_int(AUTO_BUDGET_MB_ENV, DEFAULT_AUTO_BUDGET_MB)
        if value is None
        else int(value)
    )
    if budget < 1:
        raise ValueError(
            f"{AUTO_BUDGET_MB_ENV}={budget} must be >= 1 (it is a memory envelope "
            "in MiB: pop_size * tree_len * rows * 4 bytes)"
        )
    return budget


def auto_row_allowance(
    pop_size: Optional[int] = None,
    tree_len: Optional[int] = None,
    *,
    auto_max_rows: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> int:
    """The ``auto`` policy's data-row allowance for a population of this shape.

    The rule in one line::

        allowance = min( pin , floor(budget_bytes / (pop_size * tree_len * 4)) )

    where ``pin`` is :func:`resolve_auto_max_rows` (``0`` = no pin) and ``budget``
    is :func:`resolve_auto_budget_mb`.  The bytes term is the memory envelope of
    the route: ``pop_size * tree_len * rows`` float32 values is the shape of
    dense's largest intermediate (:func:`dense_peak_bytes`), which is what grows
    with the data-row count, so it -- not the row count alone -- is what has to be
    bounded.  ``tree_len`` is the population's row stride (``max_tree_len``), the
    quantity the benchmark harness calls a cell's ``tree_len``.

    ``0`` means "no allowance limit" and is returned when the shape is unknown
    (``pop_size``/``tree_len`` is ``None``) *and* no pin is set; an unknown shape
    can therefore never produce the old, false decline.  Each term is read from
    its environment variable, so :func:`auto_rows_decline` and
    :func:`resolve_route` share exactly one implementation.
    """
    pin = resolve_auto_max_rows(auto_max_rows)
    if pop_size is None or tree_len is None:
        return pin
    per_row = max(1, int(pop_size)) * max(1, int(tree_len)) * 4
    budget_rows = max(1, (resolve_auto_budget_mb(auto_budget_mb) * (1 << 20)) // per_row)
    return budget_rows if pin <= 0 else min(pin, budget_rows)


def _rows_decline_reason(
    rows: Optional[int],
    *,
    mode: str,
    auto_max_rows: Optional[int],
    pop_size: Optional[int] = None,
    tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> Optional[str]:
    """The rows rule itself: why ``auto`` declines dense at ``rows`` (else ``None``).

    Private single source of the half-rule; :func:`auto_rows_decline` (public, mode
    resolved from ``eval_mode``) and :func:`resolve_route` (which already has the
    resolved mode) both go through it, so the allowance exists once.  A decline is
    reported only when the row count exceeds :func:`auto_row_allowance`; the reason
    names whichever term bound it (the user's pin or the memory envelope).
    """
    if rows is None or mode != "auto":
        return None
    allowance = auto_row_allowance(
        pop_size,
        tree_len,
        auto_max_rows=auto_max_rows,
        auto_budget_mb=auto_budget_mb,
    )
    if allowance <= 0 or int(rows) <= allowance:
        return None
    shape = (
        f"pop_size={int(pop_size)} x tree_len={int(tree_len)}"
        if pop_size is not None and tree_len is not None
        else "an unknown population shape"
    )
    pin = resolve_auto_max_rows(auto_max_rows)
    known_shape = pop_size is not None and tree_len is not None
    if pin > 0 and pin <= allowance:
        # The PIN is what bound the allowance; report what the envelope alone allows
        # (the pin can only tighten, so the envelope's own allowance is >= this one).
        envelope_rows = (
            auto_row_allowance(
                pop_size, tree_len, auto_max_rows=0, auto_budget_mb=auto_budget_mb
            )
            if known_shape
            else 0
        )
        return (
            f"auto policy: {int(rows)} data rows exceed {AUTO_MAX_ROWS_ENV}={pin}, "
            f"the explicit row ceiling of the dense route at {shape}"
            + (
                f" (the {resolve_auto_budget_mb(auto_budget_mb)} MiB envelope alone "
                f"would allow {envelope_rows} rows, so the pin is the tighter term)"
                if known_shape
                else ""
            )
            + f"; unset it (or set 0) to let {AUTO_BUDGET_MB_ENV} decide, or set "
            "EVOGP_EVAL_MODE=dense to force the dense route."
        )
    budget = resolve_auto_budget_mb(auto_budget_mb)
    return (
        f"auto policy: {int(rows)} data rows at {shape} need "
        f"{int(rows) * max(1, int(pop_size)) * max(1, int(tree_len)) * 4 / 2**30:.2f} "
        f"GiB in dense's largest intermediate (pop_size * tree_len * rows * 4 "
        f"bytes), above the {AUTO_BUDGET_MB_ENV}={budget} MiB memory envelope of the "
        f"route (at most {allowance} rows at that shape). The dense graph sizes its "
        "buffers by that product while the specialized engine's warm cost is "
        "rows-flat, so past the envelope the whole-population shape is no longer "
        f"schedulable. Raise {AUTO_BUDGET_MB_ENV}, pin a smaller ceiling with "
        f"{AUTO_MAX_ROWS_ENV}, or set EVOGP_EVAL_MODE=dense to force the dense route."
    )


def auto_rows_decline(
    rows: Optional[int],
    *,
    eval_mode: Optional[str] = None,
    auto_max_rows: Optional[int] = None,
    pop_size: Optional[int] = None,
    tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> Optional[str]:
    """Why the ``auto`` policy declines dense at ``rows`` data rows (else ``None``).

    The single source of the rows rule, and it is a MEMORY rule, not a row ceiling
    (:func:`auto_row_allowance`): dense's buffers are sized by
    ``pop_size * tree_len * rows`` while the specialized engine's warm cost is
    rows-flat, so what ``auto`` bounds is that product against
    ``EVOGP_DENSE_AUTO_BUDGET_MB`` (default :data:`DEFAULT_AUTO_BUDGET_MB` = 8192
    MiB), with ``EVOGP_DENSE_AUTO_MAX_ROWS`` as an optional, tightening row pin.

    ``pop_size`` / ``tree_len`` are the population's shape (``tree_len`` = the row
    stride, ``max_tree_len``); when either is unknown the envelope cannot be
    applied and only the pin can decline -- an unknown shape never produces a
    decline of its own, which is what keeps a rows-only caller from re-creating the
    old false ceiling.  ``rows=None`` (unknown, e.g. a ``per_row=True`` call) never
    declines.  An explicit ``EVOGP_EVAL_MODE=dense`` always wins -- the user's
    opt-in is not second-guessed, the same way an explicit ``specialized`` is never
    overridden.
    """
    return _rows_decline_reason(
        rows,
        mode=resolve_eval_mode(eval_mode),
        auto_max_rows=auto_max_rows,
        pop_size=pop_size,
        tree_len=tree_len,
        auto_budget_mb=auto_budget_mb,
    )


def resolve_auto_max_tree_len(value: Optional[int] = None) -> int:
    """Resolve the ``auto`` raise's ceiling (``EVOGP_DENSE_AUTO_MAX_TREE_LEN``)."""
    ceiling = (
        _env_int(AUTO_MAX_TREE_LEN_ENV, DEFAULT_AUTO_MAX_TREE_LEN)
        if value is None
        else int(value)
    )
    if ceiling < 1:
        raise ValueError(
            f"{AUTO_MAX_TREE_LEN_ENV}={ceiling} must be >= 1 (it is a trace-length "
            "ceiling in nodes)"
        )
    return ceiling


def trace_cap_pinned(max_tree_len: Optional[int] = None) -> bool:
    """Whether the trace cap was PINNED by the caller (so ``auto`` must not raise it).

    ``True`` for an explicit ``max_tree_len=`` argument or a non-empty
    ``EVOGP_DENSE_MAX_TREE_LEN``: a cap the user asked for is never
    second-guessed by the automatic raise, exactly like an explicit
    ``EVOGP_EVAL_MODE=dense`` is never gated by the ``auto`` rules.
    """
    if max_tree_len is not None:
        return True
    return bool(os.environ.get(MAX_TREE_LEN_ENV, "").strip())


def _envelope_tree_len(
    pop_size: Optional[int], rows: Optional[int], auto_budget_mb: Optional[int]
) -> int:
    """The trace length the MEMORY ENVELOPE alone allows at this shape (``0`` = n/a).

    The cap-axis mirror of the envelope term in :func:`auto_row_allowance`::

        floor(budget_bytes / (pop_size * rows * 4))

    i.e. the same ``pop_size * tree_len * rows * 4`` product solved for
    ``tree_len`` instead of for ``rows``.  ``rows`` defaults to ``1`` (the
    ``per_row=True`` shape, where the graph's ``N`` is one row) and ``0`` means
    "the shape is unknown, the envelope cannot be applied".
    """
    if pop_size is None:
        return 0
    per_node = (
        max(1, int(pop_size)) * max(1, 1 if rows is None else int(rows)) * 4
    )
    budget = resolve_auto_budget_mb(auto_budget_mb) * (1 << 20)
    return max(1, budget // per_node)


def auto_tree_len_allowance(
    pop_size: Optional[int] = None,
    rows: Optional[int] = None,
    *,
    auto_max_tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> int:
    """The trace length the ``auto`` raise may reach for a population of this shape.

    The rule in one line::

        allowance = min( ceiling , floor(budget_bytes / (pop_size * rows * 4)) )

    where ``ceiling`` is :func:`resolve_auto_max_tree_len` and ``budget`` is
    :func:`resolve_auto_budget_mb` -- the very same memory envelope the rows rule
    uses, solved for the trace length instead of the data-row count (the graph
    sizes its buffers by ``pop_size * tree_len * rows``, so either axis can be the
    one that has to give).  ``rows`` defaults to ``1`` when ``None``: that is the
    ``per_row=True`` shape, where ``x`` is ``(pop_size, input_len)`` and every
    tree sees one row, so ``N == 1`` is the graph's real row axis.

    ``pop_size=None`` returns the ceiling alone: without a population shape the
    envelope cannot be applied, and the term can only TIGHTEN the ceiling.
    """
    ceiling = resolve_auto_max_tree_len(auto_max_tree_len)
    if pop_size is None:
        return ceiling
    return min(ceiling, _envelope_tree_len(pop_size, rows, auto_budget_mb))


def _next_power_of_two(n: int) -> int:
    """``n`` rounded UP to the next power of two (``1`` for ``n <= 1``).

    The trace length dense compiles at is ``L = min(cap, row_len)`` and ``L`` is
    part of the graph key (:func:`_dense_executable_key` and the plan's spec
    shapes), so a cap that tracks the population's REALIZED longest tree would
    compile a fresh graph every time that length drifts -- i.e. once per generation
    of a GP loop.  Rounding the raised cap up to a power of two keeps one compiled
    graph alive across a whole band of realized lengths (``65..128``,
    ``129..256``) for a few percent more trace work per call.
    """
    return 1 << (max(1, int(n)) - 1).bit_length()


def _cap_decline_reason(
    longest: int,
    *,
    requested: Optional[int] = None,
    pop_size: Optional[int],
    rows: Optional[int],
    auto_max_tree_len: Optional[int],
    auto_budget_mb: Optional[int],
) -> str:
    """Why the ``auto`` raise cannot cover a ``longest``-node population (see the rule).

    Private half of :func:`_cap_decision`, written like :func:`_rows_decline_reason`:
    it names BOTH terms of the allowance and says which one bound it (the ceiling
    or the memory envelope), so a user knows which knob to turn.  ``requested`` is
    the ROUNDED cap the raise would have used (:func:`_next_power_of_two` of
    ``max(base, longest)``); it defaults to the rounding of ``longest`` itself,
    which is the same value on every path that can decline (there ``longest`` is
    already above the base cap).
    """
    if requested is None:
        requested = _next_power_of_two(int(longest))
    ceiling = resolve_auto_max_tree_len(auto_max_tree_len)
    budget = resolve_auto_budget_mb(auto_budget_mb)
    allowance = auto_tree_len_allowance(
        pop_size,
        rows,
        auto_max_tree_len=auto_max_tree_len,
        auto_budget_mb=auto_budget_mb,
    )
    envelope = _envelope_tree_len(pop_size, rows, auto_budget_mb)
    if pop_size is None:
        tighter = (
            f"{AUTO_MAX_TREE_LEN_ENV}={ceiling} is the tighter term (the population "
            "shape is unknown, so the envelope cannot be applied)"
        )
    elif ceiling <= envelope:
        tighter = (
            f"{AUTO_MAX_TREE_LEN_ENV}={ceiling} is the tighter term (the {budget} "
            f"MiB envelope alone would allow {envelope} nodes)"
        )
    else:
        tighter = (
            f"{AUTO_BUDGET_MB_ENV}={budget} MiB is the tighter term (the "
            f"{AUTO_MAX_TREE_LEN_ENV}={ceiling} ceiling alone would allow "
            f"{ceiling} nodes)"
        )
    shape = (
        f"pop_size={int(pop_size)} x {1 if rows is None else int(rows)} row(s)"
        if pop_size is not None
        else "a population of unknown shape"
    )
    return (
        f"auto policy: the longest tree is {int(longest)} nodes at {shape}, above "
        f"the {allowance} nodes the automatic trace-cap raise allows "
        f"({AUTO_MAX_TREE_LEN_ENV}={ceiling} ceiling of the raise, "
        f"{AUTO_BUDGET_MB_ENV}={budget} MiB ({budget / 1024:.2f} GiB) envelope of "
        f"pop_size * tree_len * rows * 4 bytes); {tighter}. The raise rounds the "
        "trace length up to the next power of two so a training loop reuses ONE "
        "compiled dense graph across generations, so the request here is the "
        f"{int(requested)}-node rounded value, which the {allowance}-node allowance "
        "does not reach. Dense's graph and its "
        "compile grow linearly in the trace length, so the raise is bounded. Raise "
        f"{AUTO_MAX_TREE_LEN_ENV} or {AUTO_BUDGET_MB_ENV}, or set an explicit "
        f"{MAX_TREE_LEN_ENV} (or max_tree_len=) above {int(longest)} together with "
        "EVOGP_EVAL_MODE=dense to force the dense route."
    )


def _cap_decision(
    longest: Optional[int],
    row_len: Optional[int],
    *,
    mode: str,
    pop_size: Optional[int] = None,
    rows: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    auto_max_tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> Tuple[int, Optional[str]]:
    """The ``auto`` trace-cap raise itself: ``(effective_cap, decline_reason)``.

    Private single source of the cap-axis half-rule (the twin of
    :func:`_rows_decline_reason`): :func:`auto_trace_cap` (public, mode resolved)
    and :func:`auto_cap_decline` (public, the reason alone) both go through it, so
    the effective cap and its refusal can never disagree.

    ``row_len`` is the population's row STRIDE (``max_tree_len``): when it already
    fits the cap, ``L = min(cap, row_len) == row_len`` covers every tree and there
    is nothing to raise.  Otherwise the cap is raised to
    ``_next_power_of_two(max(base, longest))`` -- the population's realized longest
    tree rounded UP to a power of two, so a training loop whose longest tree drifts
    keeps hitting ONE compiled graph -- provided
    :func:`auto_tree_len_allowance` reaches that rounded value; a request the
    allowance cannot cover is a decline that names both terms and the rounded
    request (no clamping, no partial raise).
    """
    base = _resolve_max_tree_len(max_tree_len)
    if (
        mode != "auto"
        or trace_cap_pinned(max_tree_len)
        or longest is None
        or int(longest) <= base
        or (row_len is not None and int(row_len) <= base)
    ):
        return base, None
    allowance = auto_tree_len_allowance(
        pop_size,
        rows,
        auto_max_tree_len=auto_max_tree_len,
        auto_budget_mb=auto_budget_mb,
    )
    requested = _next_power_of_two(max(base, int(longest)))
    if allowance < requested:
        return base, _cap_decline_reason(
            longest,
            requested=requested,
            pop_size=pop_size,
            rows=rows,
            auto_max_tree_len=auto_max_tree_len,
            auto_budget_mb=auto_budget_mb,
        )
    return requested, None


def auto_trace_cap(
    longest: Optional[int],
    row_len: Optional[int],
    *,
    eval_mode: Optional[str] = None,
    pop_size: Optional[int] = None,
    rows: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    auto_max_tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> int:
    """The trace cap a call should use: raised for an over-cap population under ``auto``.

    ``_next_power_of_two(max(base, longest))`` -- the population's realized longest
    tree rounded UP to a power of two, so a training loop whose longest tree drifts
    keeps one compiled graph -- when ``auto`` chose dense, the cap was not pinned
    and the population's realized longest tree exceeds it, and only while
    :func:`auto_tree_len_allowance` reaches that rounded value; else
    :func:`_resolve_max_tree_len` unchanged -- so every explicit mode
    (``dense``/``dynamic``/``specialized``/``batch``), a pinned cap and a
    population that already fits are all the identity.  ``row_len`` is the row
    stride; ``rows`` is the call's data-row count (``None`` = one row per tree).
    """
    return _cap_decision(
        longest,
        row_len,
        mode=resolve_eval_mode(eval_mode),
        pop_size=pop_size,
        rows=rows,
        max_tree_len=max_tree_len,
        auto_max_tree_len=auto_max_tree_len,
        auto_budget_mb=auto_budget_mb,
    )[0]


def auto_cap_decline(
    longest: Optional[int],
    *,
    eval_mode: Optional[str] = None,
    pop_size: Optional[int] = None,
    rows: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    auto_max_tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
) -> Optional[str]:
    """Why the ``auto`` raise cannot cover a ``longest``-node population (else ``None``).

    The decline half of :func:`auto_trace_cap`: a population whose longest tree
    exceeds what the ceiling/envelope allowance can raise the cap to is the one
    case where ``auto`` has picked a route it cannot serve, so
    :func:`evaluate_dense` / :func:`sr_fitness_dense` decline it with
    ``device_fallback=True`` (the DEVICE backend, not per-tree numpy, is then the
    sane fallback).  Only ``auto`` can decline -- an explicit
    ``EVOGP_EVAL_MODE=dense`` is the user's opt-in and is not second-guessed.
    """
    return _cap_decision(
        longest,
        None,
        mode=resolve_eval_mode(eval_mode),
        pop_size=pop_size,
        rows=rows,
        max_tree_len=max_tree_len,
        auto_max_tree_len=auto_max_tree_len,
        auto_budget_mb=auto_budget_mb,
    )[1]


#: Memo of the route policy's device probe (a ``torch.cuda.is_available()``-style
#: check, which is also True on ROCm).  ``None`` = not probed yet; the probe runs at
#: most once per process, is monkeypatchable and is never reached when a backend was
#: named explicitly -- ``EVOGP_EVOXC_BACKEND=numpy`` therefore forces the pure-CPU
#: route with no probe side effects at all.
_DEVICE_AVAILABLE: Optional[bool] = None


def device_available() -> bool:
    """Whether the route policy may pick a device backend (memoized probe)."""
    global _DEVICE_AVAILABLE
    if _DEVICE_AVAILABLE is None:
        _DEVICE_AVAILABLE = bool(torch.cuda.is_available())
    return _DEVICE_AVAILABLE


#: Memo of the route policy's FUSED probe -- ``triton`` importable AND a
#: torch-visible CUDA/HIP device -- exactly like :data:`_DEVICE_AVAILABLE`.
#: ``None`` = not probed yet, so the probe runs at most once per process, is
#: monkeypatchable, and is never reached when an eval mode or a backend was named
#: explicitly.  It imports NOTHING at module scope: the answer comes from
#: :func:`evogp.tree.fused_engine.fused_available`, which reads torch's device
#: flag and ``importlib.util.find_spec("triton")`` -- never ``import triton`` --
#: so a CPU box and the hermetic numpy-backend suite pay one bool and nothing else.
_FUSED_AVAILABLE: Optional[bool] = None


def fused_route_available() -> bool:
    """Whether ``auto`` may consider the fused (Triton) route (memoized probe).

    The device half is :func:`device_available` (the pinned memo above), so a
    caller that pins one probe pins the whole predicate and no extra torch call is
    made; :func:`fused_available` adds the ``triton`` half.
    """
    global _FUSED_AVAILABLE
    if _FUSED_AVAILABLE is None:
        from . import fused_engine

        _FUSED_AVAILABLE = bool(device_available() and fused_engine.fused_available())
    return _FUSED_AVAILABLE


def auto_fused_decline(
    pop_size: Optional[int],
    *,
    eval_mode: Optional[str] = None,
    rows: Optional[int] = None,
    tree_len: Optional[int] = None,
    output_len: int = 1,
) -> Optional[str]:
    """Why ``auto`` cannot reach the fused route for this call (``None`` = it can).

    The fused half of :func:`resolve_route`'s default policy, in the same shape as
    :func:`auto_rows_decline`: only ``auto`` is gated -- an explicit
    ``EVOGP_EVAL_MODE`` owns its engine and is never second-guessed, so it answers
    ``None``.  ``auto`` reaches the FUSED (Triton) route when, in this order,

    1. ``triton`` is importable AND a CUDA/HIP device is usable from torch
       (:func:`fused_route_available`, the memoized probe -- no ``triton`` import
       anywhere here),
    2. ``pop_size >= EVOGP_FUSED_MIN_POP`` (default 16, the measured flat-kernel
       crossover; ONE parser, :func:`resolve_fused_min_pop`), and
    3. the population fits the fused route's memory envelope -- the three-term
       byte rule the engine itself enforces before a launch, through its
       non-raising :func:`evogp.tree.fused_engine.envelope_decline`.

    ``rows``/``tree_len``/``output_len`` describe the data plane and are OPTIONAL
    (``output_len`` is the population's real output count, whose value buffer the
    envelope counts): with an unknown shape the envelope cannot be applied here,
    and ``evaluate_fused`` / ``sr_fitness_fused`` enforce it at call time, where a
    ``FusedUnsupported`` continues on the route's ``numpy`` fallback.  The wired
    call sites pass ``pop_size`` only, exactly as they do for the dense rows rule.
    """
    if resolve_eval_mode(eval_mode) != "auto":
        return None
    if not fused_route_available():
        if not device_available():
            return (
                "auto policy: no CUDA/HIP device usable from torch -> the fused "
                "(Triton) route needs one"
            )
        return (
            "auto policy: triton is not importable -> the fused (Triton) route is "
            "unavailable"
        )
    minimum = resolve_fused_min_pop()
    if int(pop_size) < minimum:
        return (
            f"auto policy: pop_size={int(pop_size)} < {minimum} "
            f"({FUSED_MIN_POP_ENV}) -> below the fused route's crossover"
        )
    if rows is None or tree_len is None:
        return None
    from . import fused_engine

    return fused_engine.envelope_decline(
        pop_size, tree_len, rows, out_len=output_len
    )


@dataclass(frozen=True)
class Route:
    """The route policy's decision for one call: which ENGINE, on which backend.

    ``engine`` names one of the four evaluation engines (:data:`ENGINES`):
    ``"dense"`` -> :func:`evaluate_dense` / :func:`sr_fitness_dense`,
    ``"batch"`` -> :func:`evogp.tree.batch_engine.evaluate_batch` /
    ``sr_fitness_batch`` (opt-in only), ``"fused"`` ->
    :func:`evogp.tree.fused_engine.evaluate_fused` / ``sr_fitness_fused`` (opt-in
    only), ``"specialized"`` ->
    :func:`evoxc_engine.decode_population` + ``evaluate`` / ``sr_fitness``.
    ``backend`` is always a concrete backend NAME (``"numpy"`` / ``"iree"``) that
    the call site passes explicitly to the engine it picked, so the route -- not
    an ambient environment variable -- decides where a call runs;
    ``fallback_backend`` is what the call site continues on after that engine's
    decline (:class:`DenseUnsupported` / ``BatchUnsupported`` /
    ``FusedUnsupported``).  ``compare=False`` on ``engine`` is deliberate: for the
    engines the policy had before the opt-in routes it is a *derived* view of
    ``use_dense``, so every ``Route`` equality check written against the old
    three-field record keeps its meaning.
    """

    use_dense: bool
    backend: str
    fallback_backend: str
    reason: str = ""
    engine: str = field(default="specialized", compare=False)

    def __post_init__(self):
        """Keep ``engine`` and ``use_dense`` consistent (the class invariant).

        ``use_dense == (engine == "dense")`` holds for every instance: an
        old-style ``Route(True, ...)`` (no ``engine`` given) is promoted to
        ``engine="dense"``, an explicit ``engine="dense"`` with
        ``use_dense=False`` raises ``ValueError`` (as do ``engine="batch"`` and
        ``engine="fused"`` with ``use_dense=True``), and an unknown engine name is
        rejected.
        """
        engine = str(self.engine)
        if engine not in ENGINES:
            raise ValueError(f"Route.engine={engine!r} is not one of {list(ENGINES)}")
        if engine == "batch":
            if self.use_dense:
                raise ValueError(
                    "Route(use_dense=True, engine='batch') is inconsistent: the "
                    "batch route owns the whole population, so use_dense must be False"
                )
            return
        if engine == "fused":
            if self.use_dense:
                raise ValueError(
                    "Route(use_dense=True, engine='fused') is inconsistent: the "
                    "fused route owns the whole population, so use_dense must be False"
                )
            return
        if engine == "dense":
            if not self.use_dense:
                raise ValueError(
                    "Route(use_dense=False, engine='dense') is inconsistent: "
                    "engine='dense' requires use_dense=True"
                )
            return
        # engine == "specialized": a caller that only set ``use_dense`` means dense.
        if self.use_dense:
            object.__setattr__(self, "engine", "dense")

    @property
    def use_batch(self) -> bool:
        """Whether the call site should attempt the opt-in batch route."""
        return self.engine == "batch"

    @property
    def use_fused(self) -> bool:
        """Whether the call site should attempt the opt-in fused (Triton) route."""
        return self.engine == "fused"


# --------------------------------------------------------------------------- #
# the route policy's memo
# --------------------------------------------------------------------------- #
#: The ambient environment's raw store, when CPython exposes it: ``os._Environ``
#: keeps its keys AND values as ``bytes`` (``surrogateescape``-encoded) in a plain
#: ``dict`` -- ``os.environ._data`` -- and a plain ``os.environ[...] = v`` /
#: ``del`` / ``.pop`` writes through to exactly that dict.  The route memo has to
#: read its knobs on EVERY call to stay sound, and this reader is what makes the
#: key cheaper than the derivation it guards: measured on the reference box, six
#: names cost 3.37 us through ``os.environ.get`` (0.56 us each -- it re-encodes the
#: key and goes through the mapping protocol) against 0.51 us for six ``dict.get``
#: calls with pre-encoded keys.  ``None`` when the private store is missing or is
#: not a ``dict``, in which case :func:`_route_env` answers through ``os.environ``
#: (and the key carries ``str`` values instead of ``bytes``) -- the two spellings
#: never mix inside one process, and both compare exactly as ``os.environ.get``
#: would.
_ENV_DATA: Optional[Dict[Any, Any]] = getattr(os.environ, "_data", None)
if not isinstance(_ENV_DATA, dict):
    _ENV_DATA = None
#: ``str`` -> encoded name, filled on first use (a handful of module constants).
_ENV_KEYS: Dict[str, bytes] = {}


def _route_env(name: str) -> Optional[str]:
    """One environment knob, RAW and unparsed (the route memo's own reader).

    Exactly what ``os.environ.get(name)`` answers -- an unset name is ``None``,
    which is what every resolver in this module reads as "use the default", and a
    set one is the same ``str`` (the store holds ``surrogateescape``-encoded
    ``bytes``, decoded the way ``os.environ`` decodes them) -- but through the fast
    reader when the interpreter has one (see :data:`_ENV_DATA`).  A store that is
    missing or does not follow that layout falls back to ``os.environ.get`` for that
    lookup rather than serving a silently-``None`` knob, so the two spellings can
    only differ in speed.
    """
    data = _ENV_DATA
    if data is not None:
        key = _ENV_KEYS.get(name)
        if key is None:
            key = _ENV_KEYS[name] = os.fsencode(name)
        value = data.get(key)
        if value is None:
            return None
        if type(value) is bytes:
            return value.decode("utf-8", "surrogateescape")
        if type(value) is str:  # an already-decoded store: trust it as-is
            return value
    return os.environ.get(name)


def _route_probe_needs(mode: str, explicit: str) -> Tuple[bool, bool]:
    """Which availability probes :func:`_route_decision` consults for this call.

    The memos' LAZINESS contract in one function: a probe
    (:data:`_DEVICE_AVAILABLE` / :data:`_FUSED_AVAILABLE`) may only be resolved by
    the policy itself, so the memo has to know -- BEFORE any key is built -- whether
    the answer it is about to cache depends on one.  The rules, arm by arm, are the
    arms of :func:`_route_decision`:

    * ``specialized`` and every arm with an EXPLICIT backend: no probe at all (that
      is what ``test_route_policy.py``'s isolation cases pin: an explicit mode or
      backend must answer without asking the box anything).
    * ``batch`` / ``fused`` / ``dense`` without an explicit backend: the device
      probe only.
    * ``auto`` without an explicit backend: the device probe gates the whole arm,
      and only a device answer of ``True`` reaches the fused half
      (``fused_route_available``), so the fused probe is needed exactly when the
      device probe is not already known to be ``False``.

    A needed probe that is still ``None`` means "derive fresh, memoize nothing" --
    the caller must never resolve it just to complete a key.
    """
    if explicit or mode == "specialized":
        return (False, False)
    if mode in ("batch", "fused", "dense"):
        return (True, False)
    return (True, _DEVICE_AVAILABLE is not False)


#: Memo of :func:`resolve_route`'s DECISION.  The policy is a pure function of its
#: arguments, the environment knobs it reads and the two memoized availability
#: probes, and it sits on the hottest path in the library (every
#: ``Forest.forward`` / ``batch_forward`` / ``SR_fitness`` and every ``Tree.*``
#: call asks it; a derivation costs 5.8 us of the fused route's ~37 us warm
#: ``p256-L64`` call, more than the kernel launch), so the whole decision is a table
#: lookup instead of a re-derivation.
#:
#: Soundness is by CONSTRUCTION, and it is the laziness rule that makes it hold:
#: the key carries every raw argument, every knob the arm can read and the two
#: probe VALUES the answer used (see :func:`_route_key`), and a needed probe that
#: is still unresolved is served by the policy, never memoized
#: (:func:`_route_probe_needs`).  A hit therefore proves exactly what a fresh
#: derivation would compute -- same inputs, same knobs, same already-taken probes --
#: and nothing is ever re-read lazily behind an entry.  A raise is never stored (the
#: error path is taken again on every call, with the same message), while a decline
#: IS a :class:`Route` and is served exactly like any other decision.
#:
#: Hygiene only to drop: :func:`clear_route_memo` runs from
#: ``evoxc_engine.clear_cache()``, the engine's single in-memory invalidation point.
_ROUTE_MEMO: Dict[Tuple[Any, ...], "Route"] = {}
_ROUTE_MEMO_MAX = 32
#: Hit/miss/bypass counters, in one-element lists so the hot path mutates them
#: without a ``global`` statement (the lock covers the cold store path and a
#: concurrent :func:`route_cache_stats` read, never a hit).
_ROUTE_HITS = [0]
_ROUTE_MISSES = [0]
_ROUTE_BYPASSES = [0]
_ROUTE_LOCK = threading.Lock()


def clear_route_memo() -> None:
    """Drop the route memo and zero its counters (``clear_cache()`` hygiene).

    No entry can go stale -- the key carries every input of the answer -- so this is
    the same kind of reset ``dense_memo.clear`` performs for the dense front tables:
    the counters are zeroed with the table, because a cleared memo has no hits to
    report.
    """
    with _ROUTE_LOCK:
        _ROUTE_MEMO.clear()
        _ROUTE_HITS[0] = 0
        _ROUTE_MISSES[0] = 0
        _ROUTE_BYPASSES[0] = 0


_register_cache_reset(clear_route_memo)


def route_cache_stats() -> Dict[str, int]:
    """Snapshot of the route memo (every key is ``route_*``).

    ``route_hits`` are calls served from the table, ``route_misses`` the calls that
    derived a decision (stored or not), ``route_bypasses`` the calls that could not
    be memoized AT ALL because a probe it needs is still unresolved
    (:func:`_route_probe_needs` -- the lazy arms), ``route_entries`` the live entry
    count and ``route_maxsize`` its bound (the table is cleared as a whole when it
    is reached).  ``evoxc_engine.clear_cache()`` resets the table through
    :func:`clear_route_memo`; the counters are reset there too, since a cleared memo
    has no hits to report.
    """
    with _ROUTE_LOCK:
        return {
            "route_hits": _ROUTE_HITS[0],
            "route_misses": _ROUTE_MISSES[0],
            "route_bypasses": _ROUTE_BYPASSES[0],
            "route_entries": len(_ROUTE_MEMO),
            "route_maxsize": _ROUTE_MEMO_MAX,
        }


def _route_key(
    pop_size: Any,
    rows: Any,
    backend: Any,
    iree_device_index: Any,
    eval_mode: Any,
    auto_min_pop: Any,
    auto_max_rows: Any,
    tree_len: Any,
    auto_budget_mb: Any,
    output_len: Any,
    mode: str,
    explicit: str,
    probes: Tuple[Any, Any],
) -> Optional[Tuple[Any, ...]]:
    """The route memo's key: every argument plus every knob the arm can read.

    The knob list is the policy chain's whole environment surface -- this module's
    five (:data:`EVAL_MODE_ENV`, ``evoxc_engine.BACKEND_ENV``,
    :data:`AUTO_MIN_POP_ENV`, :data:`AUTO_MAX_ROWS_ENV`,
    :data:`AUTO_BUDGET_MB_ENV`), the fused route's minimum population
    (:data:`FUSED_MIN_POP_ENV`) and, on the arms that reach the fused envelope (a
    ``rows`` argument) those three (:data:`FUSED_MAX_SCRATCH_MB_ENV`,
    :data:`FUSED_BLOCK_ROWS_ENV`, :data:`FUSED_NUM_WARPS_ENV`), plus the two IREE
    names ``evoxc_engine._resolve_backend`` reads (only for a named backend: an
    explicit ``backend=`` or a non-empty ``EVOGP_EVOXC_BACKEND``).  ANY new
    environment read on this path has to be added here -- that is the memo's whole
    contract, and ``tests/test_route_memo.py`` flips each name in turn.

    ``probes`` is the ``(device, fused)`` VALUE snapshot the answer used
    (:func:`_route_probe_needs` -- an unused half stays ``None``), and ``mode`` /
    ``explicit`` are the resolved spellings of the two inputs the arm is chosen on.
    ``None`` for an argument the memo cannot hash (an unhashable value): the caller
    then takes the full path and stores nothing.
    """
    env = (
        _route_env(EVAL_MODE_ENV),
        _route_env(BACKEND_ENV),
        _route_env(AUTO_MIN_POP_ENV),
        _route_env(AUTO_MAX_ROWS_ENV),
        _route_env(AUTO_BUDGET_MB_ENV),
        _route_env(FUSED_MIN_POP_ENV),
    )
    if rows is not None:
        env += (
            _route_env(FUSED_MAX_SCRATCH_MB_ENV),
            _route_env(FUSED_BLOCK_ROWS_ENV),
            _route_env(FUSED_NUM_WARPS_ENV),
        )
    if backend or env[1]:
        env += (_route_env(IREE_DEVICE_ENV), _route_env(IREE_OPT_LEVEL_ENV))
    key = (
        pop_size,
        rows,
        backend,
        iree_device_index,
        eval_mode,
        auto_min_pop,
        auto_max_rows,
        tree_len,
        auto_budget_mb,
        output_len,
        mode,
        explicit,
        probes[0],
        probes[1],
    ) + env
    try:
        hash(key)
    except TypeError:  # an unhashable argument: serve it the slow way
        return None
    return key


def _route_decision(
    pop_size: int,
    *,
    mode: str,
    explicit: str,
    rows: Optional[int] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    auto_min_pop: Optional[int] = None,
    auto_max_rows: Optional[int] = None,
    tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
    output_len: int = 1,
) -> Route:
    """The route POLICY, un-memoized (the rules are documented on :func:`resolve_route`).

    ``mode`` / ``explicit`` are the already-resolved spellings of
    ``EVOGP_EVAL_MODE`` / the explicit backend (:func:`resolve_route` resolves them
    once because its memo key needs them too, and passing them in keeps this
    function a pure function of its arguments -- no knob is read twice per call).
    """
    if mode == "specialized":
        name = resolve_backend_name(backend, iree_device_index)
        return Route(False, name, name, "EVOGP_EVAL_MODE=specialized: specialized engine only")

    if mode == "batch":
        # Opt-in third engine: no pop/rows gate (the batch engine declines a small
        # population itself), and a per-tree iree compile can never amortize, so the
        # fallback is always specialized+numpy.
        if explicit:
            name = resolve_backend_name(backend, iree_device_index)
            return Route(
                False,
                name,
                name,
                "EVOGP_EVAL_MODE=batch: always attempt the batch (merged-graph) "
                f"evaluator on the explicit {name!r} backend",
                engine="batch",
            )
        if device_available():
            return Route(
                False,
                "iree",
                "numpy",
                "EVOGP_EVAL_MODE=batch: always attempt the batch (merged-graph) "
                "evaluator on iree (device available); the fallback is specialized "
                "on numpy (a per-tree iree compile cannot amortize)",
                engine="batch",
            )
        return Route(
            False,
            "numpy",
            "numpy",
            "EVOGP_EVAL_MODE=batch: always attempt the batch (merged-graph) "
            "evaluator on numpy (no CUDA/HIP device usable from torch)",
            engine="batch",
        )
    if mode == "fused":
        # Opt-in FOURTH engine: the Triton kernel route.  It runs on the TORCH
        # device rather than on an evoxc backend, so it uses the same "iree"
        # backend spelling the other device routes use (the facade declines for any
        # other name), and its fallback is ALWAYS specialized+numpy -- the explicit
        # arm included: a per-tree iree compile can never amortize after a decline
        # (the same reasoning as the batch route's non-explicit arms), so the
        # explicit spelling drives ``backend`` only, never ``fallback_backend``.
        if explicit:
            name = resolve_backend_name(backend, iree_device_index)
            return Route(False, name, "numpy",
                         "EVOGP_EVAL_MODE=fused: always attempt the fused (Triton) "
                         f"evaluator on the explicit {name!r} backend (specialized "
                         "fallback on numpy)", engine="fused")
        if not device_available():
            return Route(False, "numpy", "numpy",
                         "EVOGP_EVAL_MODE=fused: no CUDA/HIP device usable from torch "
                         "-> specialized on numpy")
        minimum = resolve_fused_min_pop()
        if int(pop_size) < minimum:
            return Route(False, "numpy", "numpy",
                         f"EVOGP_EVAL_MODE=fused: pop_size={int(pop_size)} < {minimum} "
                         f"({FUSED_MIN_POP_ENV}) -> specialized on numpy")
        return Route(False, "iree", "numpy",
                     f"EVOGP_EVAL_MODE=fused: device available and pop_size={int(pop_size)} "
                     f">= {minimum} ({FUSED_MIN_POP_ENV}) -> fused (Triton) "
                     "(specialized fallback on numpy)", engine="fused")
    if mode == "dense":
        name = (
            resolve_backend_name(backend, iree_device_index)
            if explicit
            else ("iree" if device_available() else "numpy")
        )
        return Route(
            True,
            name,
            name,
            "EVOGP_EVAL_MODE=dense: always attempt the dense evaluator"
            + ("" if explicit else " (device backend when a device is available)"),
        )

    minimum = resolve_auto_min_pop(auto_min_pop)
    decline = _rows_decline_reason(
        rows,
        mode=mode,
        auto_max_rows=auto_max_rows,
        pop_size=pop_size,
        tree_len=tree_len,
        auto_budget_mb=auto_budget_mb,
    )
    if explicit:
        # An explicit backend keeps the historical auto rule exactly.
        name = resolve_backend_name(backend, iree_device_index)
        if decline:
            return Route(False, name, name, decline)
        if name == "numpy":
            return Route(
                False, name, name, "auto policy: explicit numpy backend -> specialized"
            )
        if int(pop_size) < minimum:
            return Route(
                False,
                name,
                name,
                f"auto policy: pop_size={int(pop_size)} < {minimum} "
                f"({AUTO_MIN_POP_ENV}) -> specialized",
            )
        return Route(
            True,
            name,
            name,
            f"auto policy: explicit {name!r} backend and pop_size={int(pop_size)} "
            f">= {minimum} ({AUTO_MIN_POP_ENV}) -> dense",
        )
    # No explicit backend: the policy owns the engine and the backend.
    if not device_available():
        return Route(
            False,
            "numpy",
            "numpy",
            "auto policy: no CUDA/HIP device usable from torch -> specialized on numpy",
        )
    # 1. the FUSED (Triton) route -- the DEFAULT on a device box: ONE launch for the
    #    whole population, no evoxc graph and no per-shape compile.  Gated by the
    #    memoized availability probe, the measured min-pop and the engine's own
    #    memory envelope (:func:`auto_fused_decline`), so a triton-less box and a
    #    population the launch could not hold both fall through to the historical
    #    dense/specialized rule below unchanged.
    if (
        auto_fused_decline(
            pop_size,
            eval_mode=mode,
            rows=rows,
            tree_len=tree_len,
            output_len=output_len,
        )
        is None
    ):
        fused_min = resolve_fused_min_pop()
        return Route(
            False,
            "iree",
            "numpy",
            f"auto policy: device available, triton importable and "
            f"pop_size={int(pop_size)} >= {fused_min} ({FUSED_MIN_POP_ENV}) -> fused "
            "(Triton) on the torch device (specialized fallback on numpy)",
            engine="fused",
        )
    # 2. the dense rule, exactly as before the fused route was the default.
    if decline:
        return Route(False, "numpy", "numpy", decline)
    if int(pop_size) >= minimum:
        return Route(
            True,
            "iree",
            "numpy",
            f"auto policy: device available and pop_size={int(pop_size)} >= "
            f"{minimum} ({AUTO_MIN_POP_ENV}) -> dense on iree "
            "(specialized fallback on numpy)",
        )
    return Route(
        False,
        "numpy",
        "numpy",
        f"auto policy: pop_size={int(pop_size)} < {minimum} ({AUTO_MIN_POP_ENV}) "
        "-> specialized on numpy (a per-tree iree compile cannot amortize there)",
    )


def resolve_route(
    pop_size: int,
    *,
    rows: Optional[int] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    eval_mode: Optional[str] = None,
    auto_min_pop: Optional[int] = None,
    auto_max_rows: Optional[int] = None,
    tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
    output_len: int = 1,
) -> Route:
    """The ONE route policy: which engine, and on which backend -- MEMOIZED.

    Every wired call site (:meth:`evogp.tree.Forest.forward` /
    ``Forest.batch_forward`` / ``Forest.SR_fitness``, :meth:`evogp.tree.Tree.forward`
    / ``Tree.SR_fitness``) asks this, so the policy exists exactly once;
    :func:`should_use_dense` is the boolean view of the dense half of it.
    ``pop_size`` is the population about to be evaluated (``1`` for a ``Tree``).

    Precedence: an explicitly provided backend (the ``backend=`` argument or a
    non-empty ``EVOGP_EVOXC_BACKEND``) first, then ``EVOGP_EVAL_MODE``, then the
    device probe.

    * ``specialized`` -> never dense, on the explicitly named backend or the
      engine's own default (``numpy``).
    * ``dense`` -> always attempt dense; with no explicit backend it uses the
      device backend on a device box (a user forcing dense wants the fast dense
      route) and ``numpy`` elsewhere.
    * ``batch`` -> always attempt the OPT-IN batch route
      (:mod:`evogp.tree.batch_engine`, ONE merged graph for the whole
      population): the explicitly named backend, else ``iree`` on a device box and
      ``numpy`` elsewhere, with a specialized+``numpy`` fallback either way.  No
      ``pop_size``/``rows`` gate here -- the batch engine declines a small
      population itself (``EVOGP_BATCH_MIN_POP``) with a ``BatchUnsupported``.
    * ``fused`` -> always attempt the OPT-IN fused route
      (:mod:`evogp.tree.fused_engine`, ONE Triton launch for the whole
      population): the explicitly named backend, else the device backend
      (``iree``, the spelling every device route uses) while a device is usable
      from torch AND ``pop_size >= EVOGP_FUSED_MIN_POP``; otherwise the route IS
      specialized+``numpy`` (``engine`` stays ``"specialized"``, so the call site
      behaves exactly like the pre-existing routes).  The fallback is ALWAYS
      specialized on ``numpy`` -- the explicit-backend arm included, even when
      ``EVOGP_EVOXC_BACKEND`` names ``iree``: the fused route has no numpy twin,
      and a per-tree iree compile can never amortize after a decline.
    * ``auto`` with an explicit backend -> the historical rule: dense iff that
      backend is not ``numpy`` *and* ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP``
      *and* ``rows`` (when given) fits :func:`auto_row_allowance`.  An explicit
      backend NEVER reaches the fused route (a named backend is the user's
      choice; ``EVOGP_EVAL_MODE=fused`` is how to ask for it).
    * ``auto`` without an explicit backend -> the policy owns the engine AND the
      backend, in this order: the FUSED (Triton) route while
      :func:`auto_fused_decline` allows it -- ``triton`` importable, a device
      usable from torch, ``pop_size >= EVOGP_FUSED_MIN_POP`` (16) and the fused
      route's memory envelope satisfied -- on the ``iree`` backend SPELLING with a
      ``numpy`` fallback (that route runs on the torch device, and a per-tree iree
      compile can never amortize after a decline); else dense on ``iree`` from
      ``EVOGP_DENSE_AUTO_MIN_POP`` (64) up; else ``numpy`` for the specialized
      route -- below the thresholds, with no device, and after a
      :class:`DenseUnsupported` (a per-tree iree compile costs 271-284 ms/tree and
      can never amortize).
    ``auto`` NEVER selects the batch engine: that half of the policy is exactly
    the fused / dense / specialized rule above.

    The rows half is a MEMORY rule (:func:`auto_rows_decline`): ``rows`` is inside
    the allowance only while ``pop_size * tree_len * rows * 4`` bytes fit
    ``EVOGP_DENSE_AUTO_BUDGET_MB`` (and ``rows`` does not exceed an explicit
    ``EVOGP_DENSE_AUTO_MAX_ROWS`` pin).  ``tree_len`` is the population's row
    stride (``max_tree_len``) and is OPTIONAL: the wired call sites do not know it,
    so they pass ``pop_size`` only and :func:`evaluate_dense` /
    :func:`sr_fitness_dense` enforce the envelope where the shape IS known -- with
    an unknown ``tree_len`` this function can only apply the pin, never a
    shape-derived decline.  The fused half has the same limitation for the same
    reason: ``output_len`` is the population's real output count (default ``1``)
    and only enters that route's envelope, which is likewise applied here only
    when the shape was passed in.

    The DECISION IS REMEMBERED (:data:`_ROUTE_MEMO`), because this function is on
    the hottest path in the library -- every ``Forest.forward`` /
    ``batch_forward`` / ``SR_fitness`` and every ``Tree.*`` call asks it -- where a
    full derivation costs 5.8 us of the fused route's ~37 us warm ``p256-L64`` call,
    i.e. more than the kernel launch, while the answer is the SAME on every call of
    a loop.  A hit proves exactly what a fresh derivation would compute, and the two
    rules that make that true are:

    * LAZINESS.  Building or validating a key never resolves a probe.  The two
      memoized availability probes are resolved by the policy alone, and only for
      the arms that need them (:func:`_route_probe_needs`: an explicit
      ``EVOGP_EVAL_MODE`` or an explicit backend answers without asking the box
      anything, which is also what keeps ``test_route_policy.py``'s isolation
      cases true).  A needed probe that is still ``None`` -- and the key's whole
      point is that a memo entry must not be built "as if" it were resolved --
      means the call is a BYPASS: the policy derives it and NOTHING is stored
      (``route_bypasses`` in :func:`route_cache_stats`).  Never an approximation of
      the probe state.
    * COMPLETENESS WITHIN AN ARM.  The key (:func:`_route_key`) carries every raw
      argument, the resolved ``mode`` / ``explicit`` spelling the arm is chosen on,
      the probe VALUES the answer used, and every environment name the arm can
      read: ``EVOGP_EVAL_MODE``, ``EVOGP_EVOXC_BACKEND``,
      ``EVOGP_DENSE_AUTO_MIN_POP``, ``EVOGP_DENSE_AUTO_MAX_ROWS``,
      ``EVOGP_DENSE_AUTO_BUDGET_MB``, ``EVOGP_FUSED_MIN_POP``, the fused envelope's
      ``EVOGP_FUSED_MAX_SCRATCH_MB`` / ``_BLOCK_ROWS`` / ``_NUM_WARPS`` (when a
      shape was passed) and ``EVOGP_EVOXC_DEVICE`` / ``EVOGP_EVOXC_OPT_LEVEL``
      (when a backend name is in play).  A new environment read on this path must
      be added there; ``tests/test_route_memo.py`` flips each name in turn and
      compares against a fresh derivation.

    Everything else is unchanged: the same ``Route`` object semantics (frozen, so a
    caller cannot mutate a memoized decision), the same ``reason`` text, and the
    same error path -- a raise is never stored, so a malformed argument or an
    unusable knob raises on every call, and a DECLINE is an ordinary ``Route`` that
    is served exactly like any other decision.  :func:`clear_route_memo` (reached
    from ``evoxc_engine.clear_cache()``) is hygiene, not correctness: no entry can
    go stale.
    """
    mode = resolve_eval_mode(
        eval_mode if eval_mode is not None else (_route_env(EVAL_MODE_ENV) or "")
    )  # raises on a bad spelling, memo or not (nothing is stored for a raise)
    raw_backend = _route_env(BACKEND_ENV)
    explicit = (
        str(backend).strip() if backend else (raw_backend or "").strip()
    )
    need_device, need_fused = _route_probe_needs(mode, explicit)
    device, fused = _DEVICE_AVAILABLE, _FUSED_AVAILABLE
    if (need_device and device is None) or (need_fused and fused is None):
        # Laziness: this arm's answer depends on a probe nobody has taken yet, and
        # the memo must never take one just to complete a key.
        _ROUTE_BYPASSES[0] += 1
        return _route_decision(
            pop_size,
            mode=mode,
            explicit=explicit,
            rows=rows,
            backend=backend,
            iree_device_index=iree_device_index,
            auto_min_pop=auto_min_pop,
            auto_max_rows=auto_max_rows,
            tree_len=tree_len,
            auto_budget_mb=auto_budget_mb,
            output_len=output_len,
        )
    probes = (device if need_device else None, fused if need_fused else None)
    key = _route_key(
        pop_size,
        rows,
        backend,
        iree_device_index,
        eval_mode,
        auto_min_pop,
        auto_max_rows,
        tree_len,
        auto_budget_mb,
        output_len,
        mode,
        explicit,
        probes,
    )
    if key is not None:
        route = _ROUTE_MEMO.get(key)
        if route is not None:
            _ROUTE_HITS[0] += 1
            return route
    _ROUTE_MISSES[0] += 1
    route = _route_decision(
        pop_size,
        mode=mode,
        explicit=explicit,
        rows=rows,
        backend=backend,
        iree_device_index=iree_device_index,
        auto_min_pop=auto_min_pop,
        auto_max_rows=auto_max_rows,
        tree_len=tree_len,
        auto_budget_mb=auto_budget_mb,
        output_len=output_len,
    )
    if key is not None:
        with _ROUTE_LOCK:
            if len(_ROUTE_MEMO) >= _ROUTE_MEMO_MAX:
                _ROUTE_MEMO.clear()
            _ROUTE_MEMO[key] = route
    return route


def should_use_dense(
    pop_size: int,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    eval_mode: Optional[str] = None,
    auto_min_pop: Optional[int] = None,
    rows: Optional[int] = None,
    auto_max_rows: Optional[int] = None,
    tree_len: Optional[int] = None,
    auto_budget_mb: Optional[int] = None,
    output_len: int = 1,
) -> bool:
    """The ONE policy decision: try the dense evaluator for this population?

    The boolean view of the dense half of :func:`resolve_route` (it delegates, so
    the rule exists exactly once): ``True`` means the call site attempts the dense
    evaluator with ``resolve_route(...).backend`` and, on
    :class:`DenseUnsupported`, continues on the specialized engine with
    ``resolve_route(...).fallback_backend``.  It is ``False`` for the ``batch``
    engine as well, and for the ``fused`` engine (those routes are reported by
    ``Route.engine`` / ``Route.use_batch`` / ``Route.use_fused``, never by this
    flag).  ``pop_size`` is the population about to be evaluated (``1`` for a
    ``Tree``); ``rows``/``tree_len`` describe the data plane and are optional --
    without ``tree_len`` the rows rule can only apply the
    ``EVOGP_DENSE_AUTO_MAX_ROWS`` pin, and the memory envelope is enforced by
    :func:`evaluate_dense` / :func:`sr_fitness_dense`, which know the real shape.
    ``output_len`` is the population's real output count and only enters the fused
    route's envelope, so it moves nothing here.
    """
    return resolve_route(
        pop_size,
        rows=rows,
        backend=backend,
        iree_device_index=iree_device_index,
        eval_mode=eval_mode,
        auto_min_pop=auto_min_pop,
        auto_max_rows=auto_max_rows,
        tree_len=tree_len,
        auto_budget_mb=auto_budget_mb,
        output_len=output_len,
    ).use_dense


def decline_backend(route: Route, exc: BaseException) -> str:
    """The backend a call site continues on after a :class:`DenseUnsupported`.

    ``route.fallback_backend`` (specialized+numpy) for every ordinary decline, and
    the ENGINE'S OWN backend (:attr:`Route.backend`) when the exception carries
    ``device_fallback`` -- the ``auto`` trace-cap/envelope decline, which happens
    only on a population too long for the trace cap, i.e. exactly where a per-tree
    numpy compile is the catastrophic answer (857.22 ms against 105.62 ms for dense
    at the raised cap, at 123-node trees / pop 2 048 / ``N=1024``) and the device
    backend is the only sane fallback.  On a CPU box ``route.backend`` IS numpy, so
    nothing changes there; an explicit ``EVOGP_EVAL_MODE=dense``/``dynamic`` or
    ``specialized`` never produces such a decline either.
    """
    if getattr(exc, "device_fallback", False) and route.backend != "numpy":
        return route.backend
    return route.fallback_backend


def resolve_update_mode(value: Optional[str], backend_name: Optional[str]) -> str:
    """Resolve ``EVOGP_DENSE_UPDATE_MODE`` (explicit argument wins)."""
    mode = value if value else os.environ.get(UPDATE_MODE_ENV, "").strip()
    if not mode:
        mode = DEFAULT_UPDATE_MODE
    mode = mode.strip().lower()
    if mode not in UPDATE_MODES:
        raise ValueError(
            f"unsupported update_mode {mode!r}: evogp.tree.dense_engine supports "
            f"{list(UPDATE_MODES)} (set {UPDATE_MODE_ENV!r} or pass update_mode=)"
        )
    if mode == "scatter" and backend_name == "iree":
        allowed = os.environ.get(ALLOW_UNSAFE_SCATTER_ENV, "").strip() not in ("", "0")
        if not allowed:
            # evoxc's iree scatter mutates its input tensor in place and the value
            # buffer is retained across the trace-time loop, so a later invoke
            # would re-apply the row writes.
            raise DenseUnsupported(
                "update_mode='scatter' is unsafe on the iree backend (evoxc's "
                "scatter mutates its input tensor in place); use "
                f"{UPDATE_MODE_ENV}=grow|concat or set "
                f"{ALLOW_UNSAFE_SCATTER_ENV}=1 to opt in for experiments"
            )
    return mode


#: Accepted spellings of :data:`PRUNE_ENV` (case-insensitive); empty = default.
_PRUNE_TRUE = ("1", "true", "yes", "on")
_PRUNE_FALSE = ("0", "false", "no", "off")


def resolve_prune(value: Optional[bool] = None) -> bool:
    """Resolve ``EVOGP_DENSE_PRUNE`` (explicit argument wins).

    Read per call, like every other knob here.  Unset/empty means
    :data:`DEFAULT_PRUNE` (pruned); ``0``/``false``/``no``/``off`` restores the
    full candidate chain byte-for-byte; anything else raises ``ValueError``
    naming the variable.
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(PRUNE_ENV, "").strip().lower()
    if not raw:
        return DEFAULT_PRUNE
    if raw in _PRUNE_TRUE:
        return True
    if raw in _PRUNE_FALSE:
        return False
    raise ValueError(
        f"unsupported {PRUNE_ENV}={raw!r}: dense_engine accepts "
        f"{list(_PRUNE_TRUE + _PRUNE_FALSE)} or an empty value"
    )

# ---------------------------------------------------------------------------
# host preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensePrep:
    """Host preprocessing of one population, ready to be fed as graph inputs.

    Every array is ``(trace_len, pop_size)`` in prefix (pre-order) order; only
    positions ``i < tree_len[p]`` are meaningful and are marked by ``opcode !=
    OP_INACTIVE``.  The tail positions carry ``opcode == OP_INACTIVE`` and can
    never influence a real value: validation guarantees every child of a real
    node lies inside its own subtree, and padding rows never point outside
    ``[0, L)`` (clamped), so they only ever read other padding rows.
    """

    input_len: int
    output_len: int
    pop_size: int
    #: length of the population tensors' rows (``max_tree_len``)
    row_len: int
    #: ``L``: positions the graph unrolls (``min(cap, row_len)``)
    trace_len: int
    #: ``(pop_size,)`` declared tree lengths (``batch_subtree_size[:, 0]``)
    tree_len: np.ndarray
    #: ``(L, P)`` bool: position ``i`` belongs to tree ``p``
    real: np.ndarray
    opcode: np.ndarray
    func_id: np.ndarray
    const_value: np.ndarray
    var_index: np.ndarray
    child1: np.ndarray
    child2: np.ndarray
    child3: np.ndarray
    #: Canonical (sorted, unique) tuple of the function ids the population's
    #: DECODED function nodes carry, restricted to the candidate range
    #: (:data:`FULL_FUNC_IDS`): the ids the per-position candidate chain has to
    #: emit for this population, an empty tuple for a population without function
    #: nodes.  An id the chain does not emit leaves the node's value at 0.0, which
    #: is exactly what the full chain's unmatched ``select`` chain yields.
    used_func_ids: Tuple[int, ...] = ()

    @property
    def max_tree_length(self) -> int:
        """Longest declared tree in this population."""
        return int(self.tree_len.max(initial=0))

#: Set once the over-cap decline has warned (see :func:`_warn_trace_cap`): the
#: warning is emitted at most ONCE per process, so a training loop that keeps
#: producing over-cap populations is not flooded.  Only the warning is deduped --
#: the :class:`DenseUnsupported` fallback itself happens on every call.
_TRACE_CAP_WARNED = False


def _warn_trace_cap(longest: int, cap: int, row_len: int, backend_name: str) -> None:
    """Warn ONCE per process about the one *expensive* decline: an over-cap tree.

    Every other :class:`DenseUnsupported` decline stays silent by design: the
    specialized engine is the cheap (or the only correct) answer there --
    multi-output, a structural doubt, the ``auto`` rows rule -- and numpy is
    cheap there too.  An over-cap tree on a NON-numpy backend is the exception:
    the fallback compiles one graph PER DISTINCT TREE STRUCTURE (~0.5-2.6 s
    each, i.e. up to ``pop_size`` compiles for a fresh population) where the
    dense route would have paid ONE compile for the whole population, so the
    user should know the cap ``EVOGP_DENSE_MAX_TREE_LEN`` is what did it.
    ``backend_name`` is the already-resolved backend; the caller skips numpy
    before calling.  The flag is process-global: the first over-cap population on
    a device backend warns, later ones do not.
    """
    global _TRACE_CAP_WARNED
    if _TRACE_CAP_WARNED:
        return
    _TRACE_CAP_WARNED = True
    warnings.warn(
        f"the dense evaluator declined a population whose longest tree is "
        f"{longest} nodes because it exceeds the trace cap L={cap} "
        f"({MAX_TREE_LEN_ENV}={cap}, row length {row_len}); the call falls back "
        f"to the specialized engine on the {backend_name!r} backend, which "
        "compiles one graph PER DISTINCT TREE STRUCTURE (well under a second on "
        "numpy, but ~0.5-2.6 s each on a device backend, i.e. up to pop_size "
        "compiles for a fresh population) instead of ONE graph for the whole "
        f"population. Raise {MAX_TREE_LEN_ENV} to include these trees (one dense "
        "compile per shape signature: ~13.9 s at L=32 and ~31.6 s at L=64 on "
        "iree; see DENSE_EVAL.md §4/§9). This is reported once per process.",
        RuntimeWarning,
        stacklevel=3,
    )


def prepare_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    *,
    max_tree_len: Optional[int] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> DensePrep:
    """Decode/validate a population for the dense graph (host-side, vectorized).

    Validation is a **superset** of :func:`evoxc_engine.decode_population`'s
    structural checks, applied only to positions that belong to a tree: a
    population the dense path accepts is one the specialized decoder accepts
    too, and anything doubtful raises :class:`DenseUnsupported`.

    ``backend`` / ``iree_device_index`` are used for ONE thing: deciding whether
    the over-cap decline is worth a warning (:func:`_warn_trace_cap`).  They
    change nothing about the decode or the raised ``DenseUnsupported``.

    Raises:
        DenseUnsupported: unsupported ``output_len``, an empty population, a tree
            longer than ``L = min(max_tree_len, row_len)``, an OUT_NODE tag, or
            any malformed structure (subtree size / child position / variable
            index / function id).
        TypeError, ValueError: the same shape/type checks the specialized
            decoder performs on its inputs (2-D tensors, matching shapes,
            ``input_len > 0``).
    """
    input_len = int(input_len)
    output_len = int(output_len)
    if input_len <= 0:
        raise ValueError(f"input_len={input_len} must be positive")
    if output_len <= 0:
        raise ValueError(f"output_len={output_len} must be positive")
    if output_len != 1:
        raise DenseUnsupported(
            f"output_len={output_len}: the dense evaluator implements the "
            "single-output protocol only"
        )
    values = _as_numpy(batch_node_value, np.float32, "batch_node_value")
    types = _as_numpy(batch_node_type, np.int16, "batch_node_type")
    sizes = _as_numpy(batch_subtree_size, np.int16, "batch_subtree_size")
    if values.ndim != 2:
        raise TypeError(
            "batch_node_value must be 2-D (pop_size, max_tree_len), got shape "
            f"{values.shape}"
        )
    if types.shape != values.shape or sizes.shape != values.shape:
        raise ValueError(
            "batch_node_value/batch_node_type/batch_subtree_size must have the "
            f"same shape, got {values.shape}, {types.shape}, {sizes.shape}"
        )
    pop_size, row_len = int(values.shape[0]), int(values.shape[1])
    if pop_size <= 0 or row_len <= 0:
        raise DenseUnsupported(f"empty population shape {values.shape}")

    cap = _resolve_max_tree_len(max_tree_len)
    L = min(cap, row_len)
    tree_len = sizes[:, 0].astype(np.int32)
    if (tree_len <= 0).any():
        raise DenseUnsupported("non-positive subtree_size[0] (tree length)")
    longest = int(tree_len.max())
    if longest > L:
        # The ONE expensive decline: report it (once per process, device backends
        # only -- see _warn_trace_cap).  The fallback itself is unchanged.
        backend_name = resolve_backend_name(backend, iree_device_index)
        if backend_name != "numpy":
            _warn_trace_cap(longest, cap, row_len, backend_name)
        raise DenseUnsupported(
            f"longest tree is {longest} nodes but the dense trace cap is {L} "
            f"({MAX_TREE_LEN_ENV}={cap}, row length {row_len}); raise the cap or "
            "fall back to the specialized engine"
        )
    # The OUT_NODE rule (``multi_output.check_multi_output``): the same one the
    # specialized decoder enforces node by node, applied here to the whole table
    # at once -- over the LIVE cells only (``tree_len``), because padding behind a
    # tree may hold garbage types.  ``DenseUnsupported`` (not the shared
    # ``MultiOutputError``) so ``auto``/``dense`` DECLINE into the fallback chain,
    # which then raises the specialized decoder's ``ValueError`` for this population.
    check_multi_output(types, output_len, row_lengths=tree_len, error=DenseUnsupported)
    columns = _decode_columns(
        np.ascontiguousarray(values[:, :L].T),
        np.ascontiguousarray(types[:, :L].T),
        np.ascontiguousarray(sizes[:, :L].T).astype(np.int32),
        tree_len,
        input_len,
        L,
    )
    rows = np.arange(L, dtype=np.int32)[:, None]
    real = np.ascontiguousarray(rows < tree_len[None, :])
    # -- used function ids (host-side union over every function node) -----------
    function_nodes = real & (columns[0] >= OP_UFUNC)
    ids = np.unique(columns[1][function_nodes])
    used_func_ids = tuple(int(v) for v in ids if 1 <= int(v) <= _MAX_FUNC_ID)
    return DensePrep(
        input_len=input_len,
        output_len=output_len,
        pop_size=pop_size,
        row_len=row_len,
        trace_len=L,
        tree_len=tree_len,
        real=real,
        opcode=columns[0],
        func_id=columns[1],
        const_value=columns[2],
        var_index=columns[3],
        child1=columns[4],
        child2=columns[5],
        child3=columns[6],
        used_func_ids=used_func_ids,
    )

def _decode_columns(
    val_t: np.ndarray,
    type_t: np.ndarray,
    size_t: np.ndarray,
    tree_len: np.ndarray,
    input_len: int,
    L: int,
) -> Tuple[np.ndarray, ...]:
    """Vectorized node decode + structural validation of ``(L, P)`` node tables.

    The structural check is the vectorized form of the specialized decoder's
    reverse-prefix pass: for every node inside a tree, ``subtree_size`` must
    equal ``1 + sum(subtree_size[child])``, the first child of node ``i`` must
    be ``i+1``, the other children follow from the declared sizes
    (``c2 = i+1+s[c1]``, ``c3 = c2+s[c2]``), every child must stay inside the
    node's own subtree, and the subtree must stay inside the declared tree.  A
    valid prefix encoding is then exactly the stack order the specialized
    engine evaluates, so the child positions computed here are the real ones.

    The OUT_NODE (multi-output) rule is NOT checked here: it is a property of the
    population + ``output_len``, so :func:`prepare_dense` applies the shared
    ``multi_output.check_multi_output`` rule to the whole table before this
    decode (one rule, one message, both routes).
    """
    raw = type_t.astype(np.int32)
    base = raw & int(NType.TYPE_MASK)
    rows = np.arange(L, dtype=np.int32)[:, None]
    real = rows < tree_len[None, :]
    # 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC/unknown base (arity 3)
    op = np.where(
        base <= 1, base, np.where(base == 2, 2, np.where(base == 3, 3, OP_TFUNC))
    ).astype(np.int32)
    leaf_real = real & (op <= 1)
    fn = real & (op >= 2)
    if (size_t[leaf_real] != 1).any():
        raise DenseUnsupported("a leaf node declares subtree_size != 1")

    # -- children (prefix encoding) ------------------------------------------
    c1_raw = rows + 1
    c1 = np.broadcast_to(np.minimum(c1_raw, L - 1), size_t.shape)
    s_c1 = np.take_along_axis(size_t, c1, axis=0)
    c2_raw = c1_raw + s_c1
    c2 = np.minimum(c2_raw, L - 1)
    s_c2 = np.take_along_axis(size_t, c2, axis=0)
    c3_raw = c2_raw + s_c2
    c3 = np.minimum(c3_raw, L - 1)
    s_c3 = np.take_along_axis(size_t, c3, axis=0)
    arity = np.where(op == OP_UFUNC, 1, np.where(op == OP_BFUNC, 2, 3)).astype(
        np.int32
    )
    need2 = fn & (arity >= 2)
    need3 = fn & (arity == 3)
    expected = 1 + s_c1 + np.where(need2, s_c2, 0) + np.where(need3, s_c3, 0)
    limit = rows + size_t - 1  # last position of this node's subtree
    bad = fn & (size_t != expected)
    bad |= fn & (c1_raw > limit)
    bad |= need2 & (c2_raw > limit)
    bad |= need3 & (c3_raw > limit)
    bad |= real & (rows + size_t > tree_len[None, :])
    if bad.any():
        raise DenseUnsupported(
            "malformed tree structure (subtree sizes / child positions are "
            "inconsistent)"
        )

    # -- leaves ---------------------------------------------------------------
    shape = size_t.shape
    opcode = np.where(real, op, OP_INACTIVE).astype(np.int32)
    func_id = np.zeros(shape, dtype=np.int32)
    const_value = np.zeros(shape, dtype=np.float32)
    var_index = np.zeros(shape, dtype=np.int32)
    var_real = real & (op == OP_VAR)
    if var_real.any():
        raw_var = val_t[var_real]
        if not np.isfinite(raw_var).all():
            raise DenseUnsupported("a non-finite variable index")
        indices = np.trunc(raw_var.astype(np.float64)).astype(np.int64)
        if (indices < 0).any() or (indices >= input_len).any():
            raise DenseUnsupported(f"a variable index outside [0, {input_len})")
        var_index[var_real] = indices.astype(np.int32)
    const_real = real & (op == OP_CONST)
    const_value[const_real] = val_t[const_real]

    # -- function ids (truncate toward zero, clamp into the dispatch range) ---
    if fn.any():
        raw_fn = val_t[fn].astype(np.float64)
        if not np.isfinite(raw_fn).all():
            raise DenseUnsupported("a non-finite function id")
        truncated = np.trunc(raw_fn)
        func_id[fn] = np.where(
            truncated < 0, 0, np.minimum(truncated, float(_FUNC_CLAMP))
        ).astype(np.int32)

    # -- child positions: in range everywhere (a padding row reads garbage, but
    #    it must never produce an out-of-range index -- iree poisons OOB) ------
    children = tuple(
        np.where(real, np.minimum(c, L - 1), 0).astype(np.int32) for c in (c1, c2, c3)
    )
    return (
        np.ascontiguousarray(opcode),
        np.ascontiguousarray(func_id),
        np.ascontiguousarray(const_value),
        np.ascontiguousarray(var_index),
        np.ascontiguousarray(children[0]),
        np.ascontiguousarray(children[1]),
        np.ascontiguousarray(children[2]),
    )


# ---------------------------------------------------------------------------
# the graph (module-level: build_cached keys on id(fn))
# ---------------------------------------------------------------------------


def _col(tensor: Any, i: int, P: int) -> Any:
    """Row ``i`` of an ``(L, P)`` runtime input as a ``(P, 1)`` symbolic tensor."""
    return evoxc.reshape(evoxc.slice(tensor, (i, 0), (1, P)), (P, 1))


def _read_buffer(
    buf: Any, rows: int, child: Any, offset_base: int, P: int, N: int, PN: int, pn_off: Any
) -> Any:
    """Gather row ``child[p]`` for every tree ``p`` -> ``(P, N)``.

    ``child`` is a clamped ``(P, 1)`` int32 tensor, ``rows`` the length of the
    buffer's leading axis and ``offset_base`` the shift that turns a node index
    into a buffer row (``i+1`` for the growing buffer, ``0`` for a full one).
    The resulting offset is clamped into ``[0, rows-1]`` so padding rows can
    never produce an out-of-range index.
    """
    off = evoxc.subtract(child, offset_base)
    off = evoxc.maximum(evoxc.minimum(off, rows - 1), 0)
    index = evoxc.reshape(evoxc.add(evoxc.multiply(off, PN), pn_off), (PN,))
    flat = evoxc.reshape(buf, (rows * PN,))
    return evoxc.reshape(evoxc.gather(flat, index, axis=0), (P, N))


def _dense_body(
    opcode: Any,
    func_id: Any,
    const_value: Any,
    var_index: Any,
    child1: Any,
    child2: Any,
    child3: Any,
    x: Any,
    cfg: Sequence[Any],
) -> Any:
    """Trace-time body of the dense evaluator (``cfg`` is a static tuple).

    ``cfg = (L, P, N, input_len, per_row, update_mode, used_func_ids)`` (the loss
    variant appends its ``use_mse`` flag at index 7).  Returns the root values,
    ``(P, N)`` (``per_row=True`` uses ``N == 1``).

    ``used_func_ids`` is the population's used function ids (:func:`resolve_prune`
    on, the default): a canonical tuple the candidate chain emits ops for, in
    DESCENDING id order -- the exact order the full chain
    (:data:`FULL_FUNC_IDS`, i.e. ``EVOGP_DENSE_PRUNE=0``) walks -- so with the
    full set the emitted stream is byte-identical to the unpruned one.  It may be
    empty (a population without function nodes): no candidate op is emitted at
    all, the node value falls back to the ``zero`` the unmatched ``select`` chain
    starts from, and every traced position still has a defined value.
    """
    L = int(cfg[0])
    P = int(cfg[1])
    N = int(cfg[2])
    input_len = int(cfg[3])
    per_row = bool(cfg[4])
    mode = str(cfg[5])
    used_func_ids = tuple(int(v) for v in cfg[6])
    PN = P * N
    pn_off = evoxc.add(
        evoxc.reshape(evoxc.cast(evoxc.arange(P), evoxc.int32) * N, (P, 1)),
        evoxc.reshape(evoxc.cast(evoxc.arange(N), evoxc.int32), (1, N)),
    )
    zero = _trace_const(np.float32(0.0))
    if per_row:
        # x is (P, input_len): tree p reads its own row p.
        x_flat = evoxc.reshape(x, (P * input_len,))
        x_base = evoxc.reshape(evoxc.cast(evoxc.arange(P), evoxc.int32) * input_len, (P, 1))
    else:
        # x is (N, input_len): every tree reads all N rows.
        x_flat = evoxc.reshape(x, (N * input_len,))
        x_base = evoxc.reshape(evoxc.cast(evoxc.arange(N), evoxc.int32) * input_len, (1, N))

    if mode in ("concat", "scatter"):
        buf = evoxc.constant(evoxc.core.tensor(np.zeros((L, P, N), dtype=np.float32)))
    else:
        buf = None
    buf_rows = 0
    root = None

    for i in range(L - 1, -1, -1):
        op_i = _col(opcode, i, P)
        f_i = _col(func_id, i, P)
        cv_i = _col(const_value, i, P)
        vi_i = _col(var_index, i, P)
        c1_i = _col(child1, i, P)
        c2_i = _col(child2, i, P)
        c3_i = _col(child3, i, P)

        # -- child values ------------------------------------------------------
        if buf is None:
            # grow mode at i == L-1: a function node here would need a child
            # outside the row, which decode-time validation rejects, so the three
            # gathers are dead code and are replaced by a constant.
            v1 = v2 = v3 = _trace_const(np.zeros((P, 1), dtype=np.float32))
            has_children = False
        elif mode == "grow":
            v1 = _read_buffer(buf, buf_rows, c1_i, i + 1, P, N, PN, pn_off)
            v2 = _read_buffer(buf, buf_rows, c2_i, i + 1, P, N, PN, pn_off)
            v3 = _read_buffer(buf, buf_rows, c3_i, i + 1, P, N, PN, pn_off)
            has_children = True
        else:
            v1 = _read_buffer(buf, L, c1_i, 0, P, N, PN, pn_off)
            v2 = _read_buffer(buf, L, c2_i, 0, P, N, PN, pn_off)
            v3 = _read_buffer(buf, L, c3_i, 0, P, N, PN, pn_off)
            has_children = True

        # -- leaf value --------------------------------------------------------
        xidx = evoxc.reshape(evoxc.add(vi_i, x_base), (PN,))
        xv = evoxc.reshape(evoxc.gather(x_flat, xidx, axis=0), (P, N))
        leaf = evoxc.select(evoxc.equal(op_i, OP_VAR), xv, cv_i)

        if not has_children:
            val = leaf
        else:
            # -- function value: the candidate chain, engine emitters -------------
            # Only the used ids are emitted (descending, like the full chain); an
            # id that is emitted nowhere leaves ``prod`` at ``zero``.
            prod = zero
            for fid in reversed(used_func_ids):
                candidate = (
                    _emit_binary(fid, v1, v2)
                    if fid <= _BFUNC_MAX
                    else _emit_unary(fid, v1)
                )
                prod = evoxc.select(evoxc.equal(f_i, fid), _operand(candidate), prod)
            valid = evoxc.logical_or(
                evoxc.logical_and(
                    evoxc.equal(op_i, OP_BFUNC),
                    evoxc.logical_and(
                        evoxc.greater_equal(f_i, 1), evoxc.less_equal(f_i, _BFUNC_MAX)
                    ),
                ),
                evoxc.logical_and(
                    evoxc.equal(op_i, OP_UFUNC),
                    evoxc.greater_equal(f_i, _BFUNC_MAX + 1),
                ),
            )
            res = evoxc.select(valid, prod, zero)
            res = evoxc.select(evoxc.equal(op_i, OP_TFUNC), _emit_ternary(v1, v2, v3), res)
            val = evoxc.select(evoxc.less_equal(op_i, OP_CONST), leaf, res)

        if i == 0:
            root = val
            break

        # -- write row i -------------------------------------------------------
        row = evoxc.reshape(val, (1, P, N))
        if mode == "grow":
            buf = row if buf is None else evoxc.concatenate([row, buf], axis=0)
            buf_rows += 1
        elif mode == "concat":
            pieces = []
            if i > 0:
                pieces.append(evoxc.slice(buf, (0, 0, 0), (i, P, N)))
            pieces.append(row)
            if L - i - 1 > 0:
                pieces.append(evoxc.slice(buf, (i + 1, 0, 0), (L - i - 1, P, N)))
            buf = (
                pieces[0]
                if len(pieces) == 1
                else evoxc.concatenate(pieces, axis=0)
            )
        else:  # scatter -- numpy only by default (see resolve_update_mode)
            index = evoxc.constant(evoxc.core.tensor(np.asarray([i], dtype=np.int32)))
            buf = evoxc.scatter(buf, index, row, axis=0)
    return root


@evoxc.defn
def _dense_step(
    opcode, func_id, const_value, var_index, child1, child2, child3, x, cfg
):
    """One compiled dense population pass (module-level: ``id(fn)`` is the key).

    The trailing ``cfg`` is a static tuple (it specializes the graph and
    participates in the ``build_cached`` key); every other argument is a runtime
    tensor input whose *values* never trigger a recompile.
    """
    return _dense_body(
        opcode, func_id, const_value, var_index, child1, child2, child3, x, cfg
    )


@evoxc.defn
def _dense_loss_step(
    opcode,
    func_id,
    const_value,
    var_index,
    child1,
    child2,
    child3,
    x,
    labels,
    keep,
    cfg,
):
    """One compiled dense SR-LOSS pass (module-level: ``id(fn)`` is the key).

    The same structure-as-data graph as :func:`_dense_step` (``cfg[:6]`` is
    exactly :func:`_dense_body`'s config, ``cfg[6]`` the used-function-id tuple)
    with the fitness reduction folded in: ``cfg[7]`` is the static ``use_MSE``
    flag and the graph's only output is the
    per-tree ``sum_N err`` (``(p_chunk,)``), so the SR hot path never brings a
    prediction back to the host -- see :func:`sr_fitness_dense` /
    :func:`run_loss_plan`.

    ``labels`` is this data chunk's ``(n_chunk, output_len)`` label block (the
    dense protocol is single-output, so ``output_len == 1``; anything else is
    already :class:`DenseUnsupported`) and ``keep`` a ``(1, n_chunk)`` float32
    mask (``1.0`` real data row, ``0.0`` padding row of a partial last chunk).
    The mask is applied with ``select``, NOT a multiply: ``_pad_rows`` duplicates
    row 0 into the padding slots, so a NaN/inf label there would otherwise poison
    the sum (``NaN * 0 == NaN``); ``select`` replaces those slots with an exact
    ``+0.0``.  The ``/ N`` and the ``(pop_size,)`` assembly are host-side
    (:func:`run_loss_plan`) so that ``N`` never has to enter the cache key.
    """
    N = int(cfg[2])
    use_mse = bool(cfg[7])
    values = _dense_body(
        opcode, func_id, const_value, var_index, child1, child2, child3, x, cfg
    )  # (P, N)
    zero = _trace_const(np.float32(0.0))
    one = _trace_const(np.float32(1.0))
    diff = evoxc.subtract(evoxc.reshape(labels, (1, N)), values)
    err = evoxc.multiply(diff, diff) if use_mse else evoxc.abs(diff)
    real = evoxc.equal(evoxc.reshape(keep, (1, N)), one)
    err = evoxc.select(real, err, zero)
    return evoxc.sum(err, axes=(1,))


# ---------------------------------------------------------------------------
# chunking & plan
# ---------------------------------------------------------------------------

#: live ``(P, N)`` intermediates added per slot by :func:`dense_peak_bytes`
_PN_EXTRA = _LIVE_CANDIDATES + 8


def dense_peak_bytes(
    L: int, P: int, N: int, update_mode: str = DEFAULT_UPDATE_MODE
) -> int:
    """Documented peak-memory model of one dense chunk (see the module docstring).

    ``4 * (buf_factor*L*P*N + (LIVE+8)*P*N + 4*L*P)`` float32 bytes with
    ``buf_factor = 2`` for ``grow`` (the buffer is rebuilt by concatenation, so
    one extra copy is live) and ``3`` otherwise (initial buffer + one full-size
    update temporary).
    """
    buf_factor = 2 if update_mode == "grow" else 3
    elements = buf_factor * L * P * N + _PN_EXTRA * P * N + 4 * L * P
    return int(4 * elements)


def _chunk_shape(
    L: int, P: int, N: int, update_mode: str, budget_bytes: float
) -> Tuple[int, int]:
    """Pick ``(P_chunk, N_chunk)`` so :func:`dense_peak_bytes` fits the budget."""
    if dense_peak_bytes(L, P, N, update_mode) <= budget_bytes:
        return P, N
    per_pn = 4 * ((2 if update_mode == "grow" else 3) * L + _PN_EXTRA)
    max_pn = max(1, int(budget_bytes // per_pn))
    n_chunk = max(1, min(N, max_pn // max(1, P)))
    p_chunk = P
    if P * n_chunk > max_pn:
        p_chunk = max(1, min(P, max_pn))
    return p_chunk, n_chunk


@dataclass(frozen=True)
class DensePlan:
    """Everything :func:`build_dense` / :func:`evaluate_dense` need.

    ``specs`` is the exact ``build_cached`` spec tuple of THIS plan's variant
    (runtime :class:`evoxc.TensorSpec` leaves + the static ``cfg``); ``cfg`` alone
    is what :func:`evoxc.run` must receive as the trailing static argument.
    ``variant`` picks the module-level ``@evoxc.defn`` (:func:`_dense_step` or
    :func:`_dense_loss_step`), i.e. the two variants are two distinct cache
    entries per shape signature -- and their specs never have the same length, so
    one variant can never be mistaken for the other.
    """

    prep: DensePrep
    per_row: bool
    #: the graph's ``N`` (``1`` for ``per_row=True``, else the data-point count)
    n_rows: int
    #: ``x``'s row count for the WHOLE call (``pop_size`` / data-point count)
    x_rows: int
    p_chunk: int
    n_chunk: int
    update_mode: str
    budget_mb: float
    input_len: int
    cfg: Tuple[Any, ...] = field(repr=False)
    specs: Tuple[Any, ...] = field(repr=False)
    #: :data:`VALUE_VARIANT` or :data:`LOSS_VARIANT` (see :func:`plan_dense`)
    variant: str = VALUE_VARIANT
    #: loss variant only: the static MSE/MAE selector baked into ``cfg[7]``
    use_mse: bool = True
    #: The function ids the graph emits candidates for (``cfg[6]``): the
    #: population's used ids when :func:`resolve_prune` is on, else
    #: :data:`FULL_FUNC_IDS`.  Part of the cache key through ``cfg``.
    used_func_ids: Tuple[int, ...] = FULL_FUNC_IDS
    #: whether the candidate chain was pruned (:func:`resolve_prune`)
    prune: bool = DEFAULT_PRUNE
    @property
    def trace_len(self) -> int:
        return self.prep.trace_len

    @property
    def pop_size(self) -> int:
        return self.prep.pop_size

    @property
    def p_chunks(self) -> int:
        return -(-self.pop_size // self.p_chunk)

    @property
    def n_chunks(self) -> int:
        return -(-self.n_rows // self.n_chunk)

    @property
    def peak_bytes(self) -> int:
        return dense_peak_bytes(
            self.trace_len, self.p_chunk, self.n_chunk, self.update_mode
        )


def plan_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    n_rows: int,
    *,
    per_row: bool,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
    backend: Optional[str] = None,
    variant: str = VALUE_VARIANT,
    use_mse: bool = True,
    prune: Optional[bool] = None,
) -> DensePlan:
    """Preprocess the population, pick the chunk shape and build the specs.

    ``n_rows`` is ``x``'s row count in the :func:`evoxc_engine.evaluate` sense
    (``pop_size`` when ``per_row=True``, the data-point count otherwise).  The
    graph's own ``N`` is ``1`` for ``per_row=True``; chunking then shaves ``P``.

    ``variant`` selects the compiled graph: :data:`VALUE_VARIANT` (the default,
    one ``(P, N)`` value block per chunk -- :func:`evaluate_dense`) or
    :data:`LOSS_VARIANT` (the SR loss folded in, one ``(P,)`` loss sum per chunk
    -- :func:`sr_fitness_dense`); ``use_mse`` is the loss variant's static MSE/MAE
    selector and therefore part of its cache key.  The loss variant needs
    ``per_row=False`` (the SR protocol: every tree sees all data rows) and adds
    two runtime payloads -- the chunk's ``(n_chunk, output_len)`` labels and a
    ``(1, n_chunk)`` padding mask -- which is why its spec tuple is longer.
    """
    if variant not in PLAN_VARIANTS:
        raise ValueError(
            f"unsupported plan variant {variant!r}: dense_engine supports "
            f"{list(PLAN_VARIANTS)}"
        )
    if variant == LOSS_VARIANT and per_row:
        raise ValueError(
            "the loss variant implements sr_fitness only (per_row=False): a "
            "per_row=True loss would need one label row per tree, which is not "
            "the SR contract"
        )
    prep = prepare_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        max_tree_len=max_tree_len,
        backend=backend,
    )
    # Same backend-name resolution as the runtime path (single-sourced in
    # resolve_backend_name, which delegates to evoxc_engine._resolve_backend).
    backend_name = resolve_backend_name(backend)
    mode = resolve_update_mode(update_mode, backend_name)
    budget = _resolve_budget_mb(budget_mb)
    x_rows = int(n_rows) if per_row else max(1, int(n_rows))
    if per_row:
        x_rows = prep.pop_size
    graph_n = 1 if per_row else x_rows
    p_chunk, n_chunk = _chunk_shape(
        prep.trace_len,
        prep.pop_size,
        graph_n,
        mode,
        budget * 1024.0 * 1024.0,
    )
    shape = (prep.trace_len, p_chunk)
    x_shape = (
        (p_chunk, prep.input_len) if per_row else (n_chunk, prep.input_len)
    )
    # Candidate-chain pruning (the one op-count lever): the graph emits candidate
    # ops for the population's used function ids only.  ``prune=False`` -- or
    # ``EVOGP_DENSE_PRUNE=0`` -- keeps :data:`FULL_FUNC_IDS`, the byte-identical
    # full chain.  The tuple is a static cfg entry, so it specializes the graph
    # and joins the ``build_cached`` key: a population whose used-id set changes
    # recompiles (in a GP run the union saturates quickly).
    pruned = resolve_prune(prune)
    used_func_ids = prep.used_func_ids if pruned else FULL_FUNC_IDS
    cfg = (
        prep.trace_len,
        p_chunk,
        n_chunk,
        prep.input_len,
        bool(per_row),
        mode,
        used_func_ids,
    )
    specs: Tuple[Any, ...] = (
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(shape, evoxc.float32),
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(shape, evoxc.int32),
        evoxc.TensorSpec(x_shape, evoxc.float32),
    )
    if variant == LOSS_VARIANT:
        cfg = cfg + (bool(use_mse),)
        specs = specs + (
            evoxc.TensorSpec((n_chunk, prep.output_len), evoxc.float32),
            evoxc.TensorSpec((1, n_chunk), evoxc.float32),
        )
    specs = specs + (cfg,)
    return DensePlan(
        prep=prep,
        per_row=bool(per_row),
        n_rows=graph_n,
        x_rows=x_rows,
        p_chunk=p_chunk,
        n_chunk=n_chunk,
        update_mode=mode,
        budget_mb=budget,
        input_len=prep.input_len,
        cfg=cfg,
        specs=specs,
        variant=variant,
        use_mse=bool(use_mse),
        used_func_ids=used_func_ids,
        prune=bool(pruned),
    )


def describe_plan(plan: DensePlan) -> Dict[str, Any]:
    """Flat, JSON-friendly summary of a plan (used by dense_spike_check.py)."""
    return {
        "trace_len": plan.trace_len,
        "max_tree_length": plan.prep.max_tree_length,
        "row_len": plan.prep.row_len,
        "pop_size": plan.pop_size,
        "input_len": plan.input_len,
        "per_row": plan.per_row,
        "n_rows": plan.n_rows,
        "update_mode": plan.update_mode,
        "p_chunk": plan.p_chunk,
        "n_chunk": plan.n_chunk,
        "p_chunks": plan.p_chunks,
        "n_chunks": plan.n_chunks,
        "peak_bytes": plan.peak_bytes,
        "variant": plan.variant,
        "use_mse": plan.use_mse,
        "prune": plan.prune,
        "used_func_ids": list(plan.used_func_ids),
        "candidate_ops": len(plan.used_func_ids),
    }


def _dense_executable_key(
    fn: Any,
    specs: Tuple[Any, ...],
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Memo key of one dense executable: everything ``build_cached`` keys on.

    ``(id(fn), plan.specs, backend name, device, frozen options)`` -- ``id(fn)``
    is the module-level ``@evoxc.defn`` body (``_dense_step`` / ``_dense_loss_step``,
    i.e. the variant), ``plan.specs`` is the whole traced spec tuple (the seven
    runtime columns, ``x``, the loss variant's ``labels``/``keep`` and the static
    ``cfg``), so two plans share an entry exactly when ``build_cached`` would
    share one.  On the numpy backend ``device`` is the frozen ``Device("cpu", 0)``
    and ``options`` is empty (documented as accepted-and-ignored there), so the
    effective key reduces to ``(id(fn), specs, "numpy")``.
    """
    return (
        id(fn),
        specs,
        backend_name,
        device,
        _freeze_options(options),
    )


def build_dense(
    plan: DensePlan,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> Any:
    """Build-or-hit the ONE dense executable of ``plan`` (shared compile cache).

    Mirrors :func:`evoxc_engine._executable`: the same in-repo ``build_cached``
    pattern, the same shared cache, and the same explicit backend/device/options
    resolution (the ambient ``EVOXC_*`` env cannot change it).  The runtime tensor
    inputs are keyed once on their ``TensorSpec``, so new populations of the same
    shape never recompile, and ``plan.variant`` picks the graph
    (:func:`_dense_step` or :func:`_dense_loss_step`) -- one cache entry each.

    In front of ``build_cached`` sits :data:`DENSE_EXECUTABLE_MEMO`
    (:func:`_dense_executable_key`, an :class:`evogp.tree.dense_memo.DenseMemo`
    sized like :data:`DENSE_CACHE`): a repeated build of the same plan -- what
    every warm call of the route does -- is one dict lookup instead of a
    re-derived cache key, exactly like the specialized route's
    ``_EXECUTABLE_MEMO``.  A hit returns the very executable ``build_cached``
    holds (identity: an executable is immutable and its static program is
    re-passed by value at every ``evoxc.run``), and
    :func:`evoxc_engine.clear_cache` resets the table.

    A memo miss also consults the opt-in DISK layer
    (:mod:`evogp.tree.artifact_cache`, ``EVOGP_EVOXC_ARTIFACT_DIR``): the artifact
    of this exact graph/plan/backend/device/options is loaded instead of being
    compiled (3.1-3.4 s at ``L=32``, 6.1 s at ``L=64`` on iree) and a real build is
    saved best-effort afterwards.  With the switch off both steps are one
    environment read and nothing else.
    """
    backend_name, device, options = _resolve_backend(backend, iree_device_index)
    fn = _dense_loss_step if plan.variant == LOSS_VARIANT else _dense_step
    try:
        key: Optional[Tuple[Any, ...]] = _dense_executable_key(
            fn, plan.specs, backend_name, device, options
        )
    except TypeError:  # pragma: no cover - an unhashable spec component
        key = None
    if key is not None:
        memoized = DENSE_EXECUTABLE_MEMO.lookup(key)
        if memoized is not None:
            return memoized
    name = DENSE_ARTIFACT_NAMES[plan.variant]
    artifact = artifact_cache.artifact_path(
        name, plan.specs, backend_name, device, _freeze_options(options)
    )
    if artifact is not None:
        loaded = artifact_cache.load(artifact, plan.specs, backend_name, device)
        if loaded is not None:
            if key is not None:
                DENSE_EXECUTABLE_MEMO.store(key, loaded)
            return loaded
    if backend_name == "numpy":
        executable = evoxc.build_cached(
            fn,
            *plan.specs,
            backend=backend_name,
            device=device,
            cache=DENSE_CACHE,
        )
    else:
        executable = evoxc.build_cached(
            fn,
            *plan.specs,
            backend=backend_name,
            device=device,
            cache=DENSE_CACHE,
            **options,
        )
    if artifact is not None:
        artifact_cache.save(executable, artifact)
    if key is not None:
        DENSE_EXECUTABLE_MEMO.store(key, executable)
    return executable


# ---------------------------------------------------------------------------
# runtime: chunked execution
# ---------------------------------------------------------------------------


def _decline_rows(x: Any) -> Optional[int]:
    """``x``'s row count when it is cheaply known (shape only), else ``None``.

    Used to enforce :func:`auto_rows_decline` BEFORE the host conversion of
    ``x``: a declined call must not pay a D2H (and, on a device backend, must not
    pay anything at all) just to learn that the dense route is not used.
    """
    shape = getattr(x, "shape", None)
    try:
        if shape is not None and len(shape) == 2:
            return int(shape[0])
    except (TypeError, ValueError):  # pragma: no cover - exotic shape objects
        return None
    return None


def _population_shape(batch_subtree_size: Any) -> Tuple[Optional[int], Optional[int]]:
    """``(pop_size, tree_len)`` of the population tensors from their SHAPE only.

    ``tree_len`` is the row stride (``max_tree_len``), the quantity
    :func:`auto_row_allowance` sizes the memory envelope with.  Like
    :func:`_decline_rows` this reads ``.shape`` and nothing else, so a declined
    attempt pays neither a device-to-host copy nor a host conversion; anything
    without a 2-D shape yields ``(None, None)``, which makes the envelope
    inapplicable (only the row pin can decline then).
    """
    shape = getattr(batch_subtree_size, "shape", None)
    try:
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    except (TypeError, ValueError):  # pragma: no cover - exotic shape objects
        return None, None
    return None, None


def _population_longest(batch_subtree_size: Any) -> Optional[int]:
    """The population's realized longest tree, ``int(batch_subtree_size[:, 0].max())``.

    The one host read the ``auto`` trace-cap raise needs (:func:`auto_trace_cap`):
    ``subtree_size[:, 0]`` is every tree's real length, and its maximum is what
    decides whether the population fits ``L = min(cap, row_len)``.  Memoized in
    :data:`DENSE_INPUT_MEMO` under the key
    ``("longest", :func:`dense_memo.token`)`` so the repeated evaluation of an
    unchanged population (and the ``forward``-then-``sr_fitness`` pair) does not
    pay the reduction -- and on a device backend the device sync -- again; the
    entry is dropped by :func:`evoxc_engine.clear_cache` with the rest of the
    table.  The ``"longest"`` prefix cannot collide with
    :func:`_resident_array`'s ``(str(dtype), token)`` keys, whose first element is
    a numpy dtype string (``"float32"``) and never a word like this one.

    A tensor that cannot be tokenized (a numpy population) is measured but not
    memoized, so the answer is always the measured int; nothing is caught here --
    a device read error propagates instead of being reported as "unknown".
    """
    key = dense_memo.token(batch_subtree_size)
    if key is not None:
        cached = DENSE_INPUT_MEMO.lookup(("longest", key))
        if cached is not None:
            return int(cached)
    longest = int(batch_subtree_size[:, 0].max())
    if key is not None:
        DENSE_INPUT_MEMO.store(("longest", key), longest)
    return longest


def _resident_array(tensor: Any, dtype: Any, name: str) -> np.ndarray:
    """Memoized :func:`evoxc_engine._as_numpy`: the host copy of an unchanged tensor.

    ``x`` / ``labels`` are the same tensors on every call of a training loop, so
    the per-call device-to-host copy is pure waste; :data:`DENSE_INPUT_MEMO`
    keeps the host array of a tensor keyed by :func:`dense_memo.token`
    (``data_ptr`` + shape + dtype + device + ``_version``), which a torch
    in-place write bumps and ``clear_cache()`` drops.  Anything that is not a
    torch tensor (a numpy ``x``) is simply not memoized, and every value/type
    error still comes from :func:`_as_numpy` itself, on the miss that produced
    the array.
    """
    key = dense_memo.token(tensor)
    if key is not None:
        cached = DENSE_INPUT_MEMO.lookup((str(dtype), key))
        if cached is not None:
            return cached
    array = _as_numpy(tensor, dtype, name)
    if key is not None:
        DENSE_INPUT_MEMO.store((str(dtype), key), array)
    return array


def _resident_input(tensor: Any, input_len: int, name: str) -> np.ndarray:
    """``(N, input_len)`` float32 host copy of ``tensor`` (memoized, same checks)."""
    array = _resident_array(tensor, np.float32, name)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (rows, {input_len}), got shape {array.shape}"
        )
    if array.shape[1] != input_len:
        raise ValueError(
            f"{name} has {array.shape[1]} columns but the population expects "
            f"input_len={input_len}"
        )
    return array


@dataclass
class DenseEntry:
    """One call's worth of dense state: the plan, its executable, its payloads.

    The value :data:`DENSE_PLAN_MEMO` holds, keyed on the (unchanged) population
    tensors plus the resolved plan parameters.  It bundles the three things the
    warm path would otherwise recompute:

    * ``plan`` -- the host decode (:func:`prepare_dense`), the chunk shape and
      the traced specs; a repeated evaluation of the same population is the same
      plan, bit for bit;
    * ``executable`` -- a strong reference to the compiled (or memo-hit) graph,
      so the entry never re-derives a ``build_cached`` key either;
    * ``payloads`` -- the ``staged`` memo of :func:`_iter_plan_chunks`: the host
      blocks AND the device tensors of every runtime input, re-used by
      reference.  They are only valid for the ``x`` / ``labels`` they were
      staged from, which ``payload_key`` (their :func:`dense_memo.token` pair)
      pins; a different ``x`` clears them and stages again.

    ``pins`` holds the storages of the population tensors, so the ``data_ptr``
    the key was derived from cannot be recycled while the entry lives (see
    :func:`evogp.tree.dense_memo.pin`).  The lock serializes concurrent calls
    that share one entry (they would otherwise race on the payload table).
    """

    plan: DensePlan
    executable: Any
    backend_name: str
    device: Any
    pins: Tuple[Any, ...] = ()
    payload_key: Optional[Tuple[Any, ...]] = None
    payload_pins: Tuple[Any, ...] = ()
    payloads: Dict[Any, Tuple[np.ndarray, Any]] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock, repr=False)


def _normalize_variant(variant: str, per_row: bool) -> None:
    """The variant rules :func:`plan_dense` enforces (checked before the memo)."""
    if variant not in PLAN_VARIANTS:
        raise ValueError(
            f"unsupported plan variant {variant!r}: dense_engine supports "
            f"{list(PLAN_VARIANTS)}"
        )
    if variant == LOSS_VARIANT and per_row:
        raise ValueError(
            "the loss variant implements sr_fitness only (per_row=False): a "
            "per_row=True loss would need one label row per tree, which is not "
            "the SR contract"
        )


def _entry_key(
    values: Any,
    types: Any,
    sizes: Any,
    *,
    input_len: int,
    output_len: int,
    n_rows: int,
    per_row: bool,
    variant: str,
    use_mse: bool,
    cap: int,
    budget: float,
    mode: str,
    pruned: bool,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Optional[Tuple[Any, ...]]:
    """Memo key of one :class:`DenseEntry`, or ``None`` (not identifiable).

    The RESOLVED plan parameters enter the key, not the raw arguments, so an
    ``EVOGP_DENSE_UPDATE_MODE`` / ``EVOGP_DENSE_BUDGET_MB`` / ``EVOGP_DENSE_PRUNE``
    / trace-cap change between two calls can never be served from a plan built
    under the old value.  A population tensor that cannot be tokenized (a numpy
    array) simply is not memoized.
    """
    tokens = tuple(dense_memo.token(tensor) for tensor in (values, types, sizes))
    if any(item is None for item in tokens):
        return None
    try:
        return (
            tokens,
            int(input_len),
            int(output_len),
            int(n_rows),
            bool(per_row),
            str(variant),
            bool(use_mse),
            int(cap),
            float(budget),
            str(mode),
            bool(pruned),
            backend_name,
            device,
            _freeze_options(options),
        )
    except TypeError:  # pragma: no cover - an unhashable option value
        return None


def _plan_entry(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    n_rows: int,
    *,
    per_row: bool,
    variant: str,
    use_mse: bool,
    max_tree_len: Optional[int],
    budget_mb: Optional[float],
    update_mode: Optional[str],
    backend: Optional[str],
    iree_device_index: Optional[int],
) -> DenseEntry:
    """Plan + executable + staged payloads of this (population, data) call.

    :func:`plan_dense` and :func:`build_dense` on a miss, one dict lookup on a
    hit -- the host decode (3 D2H + the whole vectorized decode/validate pass)
    and the ``build_cached`` key derivation are both skipped for a population
    that has not changed since the previous call (elites, a ``forward`` followed
    by ``sr_fitness``, a repeated benchmark).  A different population, an
    in-place mutated one (``_version`` bump), a different shape/variant/knob or
    ``clear_cache()`` is a miss, by construction.
    """
    _normalize_variant(variant, per_row)
    backend_name, device, options = _resolve_backend(backend, iree_device_index)
    key = _entry_key(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len=input_len,
        output_len=output_len,
        n_rows=n_rows,
        per_row=per_row,
        variant=variant,
        use_mse=use_mse,
        cap=_resolve_max_tree_len(max_tree_len),
        budget=_resolve_budget_mb(budget_mb),
        mode=resolve_update_mode(update_mode, backend_name),
        pruned=resolve_prune(None),
        backend_name=backend_name,
        device=device,
        options=options,
    )
    if key is not None:
        entry = DENSE_PLAN_MEMO.lookup(key)
        if entry is not None:
            return entry
    plan = plan_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        n_rows,
        per_row=per_row,
        variant=variant,
        use_mse=use_mse,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
    )
    executable = build_dense(
        plan, backend=backend, iree_device_index=iree_device_index
    )
    entry = DenseEntry(
        plan=plan,
        executable=executable,
        backend_name=backend_name,
        device=device,
        pins=dense_memo.pin((batch_node_value, batch_node_type, batch_subtree_size)),
    )
    if key is not None:
        DENSE_PLAN_MEMO.store(key, entry)
    return entry


def _entry_payloads(
    entry: DenseEntry, x: Any, labels: Any = None
) -> Optional[Dict[Any, Tuple[np.ndarray, Any]]]:
    """The staged payload table for ``x`` / ``labels``, or ``None`` (stage per call).

    Returns the entry's persistent ``staged`` dict when both payload tensors are
    identifiable (:func:`dense_memo.token`, so ``x`` / ``labels`` must be the
    CALLER's tensors, never their numpy copies); it is cleared whenever the token
    pair changes, so a payload can never outlive the data it was read from.
    The storages of ``x`` / ``labels`` are pinned while their payloads are held
    (``payload_pins``), which makes those tokens ABA-proof exactly like the
    population's pins.  A numpy ``x`` / ``labels`` yields ``None`` -- the call
    then stages privately, exactly as before the memo existed.
    """
    x_token = dense_memo.token(x)
    if x_token is None:
        return None
    label_token = dense_memo.token(labels) if labels is not None else None
    if labels is not None and label_token is None:
        return None
    key = (x_token, label_token)
    if entry.payload_key != key:
        entry.payloads.clear()
        entry.payload_key = key
        entry.payload_pins = dense_memo.pin((x, labels))
    return entry.payloads


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
    """Pad ``array`` (axis 0) up to ``rows`` rows by repeating its first row."""
    current = array.shape[0]
    if current >= rows:
        return array
    if current == 0:
        return np.zeros((rows,) + array.shape[1:], dtype=array.dtype)
    return np.concatenate(
        [array, np.repeat(array[:1], rows - current, axis=0)], axis=0
    )


def _chunk_rows(
    array: np.ndarray, start: int, count: int, rows: int
) -> np.ndarray:
    """One contiguous ``[start:start+count]`` slice of ``array``, padded to ``rows``.

    The host PREPARATION step of an ``x`` / ``labels`` chunk payload: with
    ``per_row=False`` the payload depends only on the chunk's ``n0``, with
    ``per_row=True`` only on ``p0``, so :func:`_iter_plan_chunks` memoizes it per
    key (prepared once and, on a device backend, uploaded once per call).
    """
    chunk = np.ascontiguousarray(array[start : start + count])
    return _pad_rows(chunk, rows)

#: the runtime input columns of a :class:`DensePrep`, in
#: :func:`plan_dense`'s ``specs`` order (and in ``cfg``-free argument order)
_COLUMN_NAMES = (
    "opcode",
    "func_id",
    "const_value",
    "var_index",
    "child1",
    "child2",
    "child3",
)


def _prep_column(
    prep: DensePrep, index: int, p0: int, p_chunk: int
) -> np.ndarray:
    """One ``(L, p_chunk)`` contiguous runtime column of one ``p``-chunk."""
    array = getattr(prep, _COLUMN_NAMES[index])
    chunk = array[:, p0 : p0 + p_chunk]
    return np.ascontiguousarray(_pad_rows(chunk.T, p_chunk).T)


def _prep_columns(prep: DensePrep, p0: int, p_chunk: int) -> Tuple[np.ndarray, ...]:
    """``(L, p_chunk)`` contiguous slices of every runtime input of one chunk."""
    return tuple(
        _prep_column(prep, index, p0, p_chunk) for index in range(len(_COLUMN_NAMES))
    )

def run_plan(
    plan: DensePlan,
    executable: Any,
    x: np.ndarray,
    *,
    backend_name: str,
    device: Any,
    staged: Optional[Dict[Any, Tuple[np.ndarray, Any]]] = None,
) -> np.ndarray:
    """Run every chunk of ``plan`` and assemble the ``(P, N)`` result (host).

    The result is the transposed / per-tree layout of
    :func:`evoxc_engine.evaluate`'s return value: ``(P, 1)`` rows for
    ``per_row=True``, ``(P, N)`` otherwise.

    ``staged`` is the persistent payload memo of a repeated call
    (:attr:`DenseEntry.payloads`, injected by :func:`evaluate_dense`): with it,
    an unchanged population/data pair re-uses the host blocks and the uploaded
    device tensors by reference, so a warm call does no preparation and no
    upload at all.  ``None`` stages privately, exactly as before.
    """
    prep = plan.prep
    out = np.empty((prep.pop_size, plan.n_rows), dtype=np.float32)
    for (p0, p_chunk, n0, n_chunk), raw in _iter_plan_chunks(
        plan, executable, x, backend_name=backend_name, device=device, staged=staged
    ):
        host = _to_host(raw, backend_name).reshape(plan.p_chunk, plan.n_chunk)
        if plan.per_row:
            out[p0 : p0 + p_chunk, 0] = host[:p_chunk, 0]
        else:
            out[p0 : p0 + p_chunk, n0 : n0 + n_chunk] = host[:p_chunk, :n_chunk]
    return out


def _keep_mask(n_chunk: int, n_real: int) -> np.ndarray:
    """``(1, n_chunk)`` float32 padding mask: ``1.0`` for the ``n_real`` real rows.

    The graph's last chunk is padded up to ``n_chunk`` (:func:`_pad_rows`), and
    :func:`_dense_loss_step` selects the padding slots to an exact ``+0.0`` with
    this mask, so a partial chunk reduces over its real rows only.
    """
    keep = np.zeros((1, n_chunk), dtype=np.float32)
    keep[0, : max(0, min(int(n_real), int(n_chunk)))] = 1.0
    return keep


def _to_host(result: Any, backend_name: str) -> np.ndarray:
    """Read one evoxc result back to host as a float32 numpy array."""
    if backend_name == "numpy":
        return np.asarray(result.numpy(), dtype=np.float32)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32)


def _iter_plan_chunks(
    plan: DensePlan,
    executable: Any,
    x: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    backend_name: str,
    device: Any,
    device_resident: bool = False,
    staged: Optional[Dict[Any, Tuple[np.ndarray, Any]]] = None,
):
    """Yield ``((p0, p_chunk, n0, n_chunk), raw_evoxc_result)`` for every chunk.

    This is the READBACK-FREE SEAM of the dense route: the iterator prepares and
    uploads each distinct payload **at most once per call** and never reads a
    result back, so a caller that can consume device-resident results (the
    ``build_dense_runner`` / training-loop entry point, or any caller that keeps
    the loss on the device) simply iterates and assembles itself.  The library's
    own consumers read back explicitly -- :func:`run_plan` (values) and
    :func:`run_loss_plan` (losses) -- through :func:`_to_host`.

    The loop is ``p0``-outer/``n0``-inner with ONE per-call payload memo covering
    both halves of a payload's life (the host preparation and the device upload),
    because the node columns depend only on ``p0`` and the ``x`` / ``labels`` /
    mask payloads only on ``n0``: a naive nesting re-preps and re-uploads both
    blocks per ``(p0, n0)`` chunk (measured 296 host preparations instead of 37
    for an 8x37 population in 1x1 chunks).  ``x``/``labels`` carry only the
    CHUNK's rows, so the memo's extra host memory is one full input block plus
    one full label block (and the same again as device tensors on a device
    backend) -- all bounded by the call's own inputs, never by the chunk count.

    ``labels`` is required by (and only by) the loss variant.
    ``device_resident=True`` is the seam's flag for a readback-free consumer: it
    refuses the numpy backend (a "device-resident" result would be a host array)
    and an UNCHUNKED plan is required, because only then is one raw result the
    whole block -- with chunking, a device-side assembly would need either an evoxc
    zero-copy export (``core.Tensor.__dlpack__`` raises for every payload-backed
    tensor today, and ``evoxc.from_dlpack`` is host-memory v1 only) or a second
    compiled graph fed with the chunk results (a compile per call).  Both are
    upstream/structural and deliberately NOT emulated here.

    ``staged`` is the **payload memo injection** :func:`build_dense_runner` uses
    to split "stage the inputs" (build phase) from "run" (``.run()``): pass a
    dict and the seam prepares + uploads every payload into it on the first
    iteration, so every later iteration with the SAME dict is a pure
    ``evoxc.run`` over the already-staged payloads (no preparation, no upload, no
    readback).  ``None`` (the default) keeps the original per-call behaviour: a
    private, empty memo.  The caller owns the dict; its values are
    ``(host_array, device_tensor_or_None)`` per payload key, and re-using it is
    only valid with the same ``plan``, ``x`` and ``labels``.
    """
    loss = plan.variant == LOSS_VARIANT
    if loss and labels is None:
        raise ValueError("the loss variant needs the labels payload")
    if not loss and labels is not None:
        raise ValueError("the value variant takes no labels payload")
    if device_resident:
        if backend_name == "numpy":
            raise DenseUnsupported(
                "device_resident results need a device backend: the numpy "
                "backend returns host arrays (nothing to keep resident)"
            )
        if plan.p_chunks > 1 or plan.n_chunks > 1:
            raise DenseUnsupported(
                "device_resident results need a single-chunk plan: the chunk "
                "results cannot be assembled on the device without an evoxc "
                f"zero-copy export (raise {BUDGET_MB_ENV} to stop chunking)"
            )
    to_device = backend_name != "numpy"
    #: payload key -> ``(host_array, device_tensor_or_None)``; both halves are
    #: memoized together so the preparation count and the upload count of a
    #: payload are the same number (``prepare`` runs exactly once per key).
    prepared: Dict[Any, Tuple[np.ndarray, Any]] = {} if staged is None else staged
    def payload(key: Any, prepare: Callable[..., np.ndarray], *args: Any) -> Any:
        """The chunk payload for ``key``: prepared once, uploaded once, reused."""
        entry = prepared.get(key)
        if entry is None:
            host = prepare(*args)
            entry = (
                host,
                evoxc.from_numpy(host).to(device) if to_device else None,
            )
            prepared[key] = entry
        return entry[1] if to_device else entry[0]

    prep = plan.prep
    n_chunks = 1 if plan.per_row else plan.n_chunks
    for p0 in range(0, prep.pop_size, plan.p_chunk):
        p_chunk = min(plan.p_chunk, prep.pop_size - p0)
        columns = tuple(
            payload(("column", index, p0), _prep_column, prep, index, p0, plan.p_chunk)
            for index in range(len(_COLUMN_NAMES))
        )
        for chunk_index in range(n_chunks):
            n0 = 0 if plan.per_row else chunk_index * plan.n_chunk
            n_chunk = min(plan.n_chunk, plan.n_rows - n0) if not plan.per_row else 1
            if plan.per_row:
                args = columns + (
                    payload(("x", p0), _chunk_rows, x, p0, p_chunk, plan.p_chunk),
                )
            else:
                args = columns + (
                    payload(("x", n0), _chunk_rows, x, n0, n_chunk, plan.n_chunk),
                )
                if loss:
                    args = args + (
                        payload(
                            ("labels", n0),
                            _chunk_rows,
                            labels,
                            n0,
                            n_chunk,
                            plan.n_chunk,
                        ),
                        payload(
                            ("keep", n0), _keep_mask, plan.n_chunk, n_chunk
                        ),
                    )
            args = args + (plan.cfg,)
            with _quiet():
                yield (p0, p_chunk, n0, n_chunk), evoxc.run(executable, *args)


def run_loss_plan(
    plan: DensePlan,
    executable: Any,
    x: np.ndarray,
    labels: np.ndarray,
    *,
    backend_name: str,
    device: Any,
    staged: Optional[Dict[Any, Tuple[np.ndarray, Any]]] = None,
) -> np.ndarray:
    """Run the loss variant and assemble the ``(pop_size,)`` loss (host).

    Only ``(p_chunk,)``-scale chunk sums ever cross back to the host; they are
    accumulated in float32 and divided by ``N`` (the data-point count) once, so
    the result is ``sum_N sum_out err / N`` -- the same formula, and the same
    float32 "sum then divide" order, as :func:`evoxc_engine.sr_fitness` (the
    in-graph reduction order differs, ~1e-7 relative on the fixtures).

    The returned array is FRESH (a new ``np.zeros``) and owned by the caller, so
    converting it to a torch tensor never hands out a memoized buffer; the
    pipeline mutates the tensor it gets in place (``NaN -> -inf``).

    ``staged`` is :func:`run_plan`'s payload memo (see there): the repeated
    ``SR_fitness`` of an unchanged population and dataset re-uses the prepared
    host blocks and the uploaded device tensors by reference.
    """
    if plan.variant != LOSS_VARIANT:
        raise ValueError(
            f"run_loss_plan needs a {LOSS_VARIANT!r} plan, got {plan.variant!r}"
        )
    totals = np.zeros((plan.prep.pop_size,), dtype=np.float32)
    for (p0, p_chunk, _n0, _n_chunk), raw in _iter_plan_chunks(
        plan,
        executable,
        x,
        labels,
        backend_name=backend_name,
        device=device,
        staged=staged,
    ):
        host = _to_host(raw, backend_name).reshape(plan.p_chunk)
        totals[p0 : p0 + p_chunk] += host[:p_chunk]
    return totals / np.float32(plan.n_rows)


def _input_to_numpy(tensor: Any, input_len: int, name: str) -> np.ndarray:
    """``(N, input_len)`` float32 host copy of ``tensor`` (mirrors the engine)."""
    array = _as_numpy(tensor, np.float32, name)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (rows, {input_len}), got shape {array.shape}"
        )
    if array.shape[1] != input_len:
        raise ValueError(
            f"{name} has {array.shape[1]} columns but the population expects "
            f"input_len={input_len}"
        )
    return array


def evaluate_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
) -> torch.Tensor:
    """Evaluate a whole population with ONE dense graph (see module docstring).

    Contracts (mirroring :func:`evoxc_engine.evaluate` exactly):

    * ``per_row=True``: ``x`` is ``(pop_size, input_len)``, tree ``i`` sees row
      ``i``; the result is ``(pop_size, output_len)``.
    * ``per_row=False``: ``x`` is ``(N, input_len)``, every tree is evaluated on
      all ``N`` rows; the result is ``(pop_size, N, output_len)``.
    * The result is fresh, contiguous, ``requires_grad=False`` float32 on ``x``'s
      device (``out_device`` overrides).  ``x`` and the population tensors are
      never modified.

    Host crossings: ``x`` is converted to host ONCE -- and not at all when the
    caller passes the same tensor again, which :data:`DENSE_INPUT_MEMO` serves
    -- the population's host decode and its ``build_cached`` key derivation are
    skipped entirely for an unchanged population (:func:`_plan_entry`), the
    graph inputs are uploaded once and then re-used by reference for an
    unchanged ``x`` (:func:`_entry_payloads`), and the assembled prediction is
    created as a torch tensor ONCE with no extra copy.  The chunk results still
    come back through :func:`run_plan` (one readback per chunk -- exactly one for
    the default unchunked plan).  A fully device-resident VALUE path (no
    readback, no H2D at all) is **blocked on evoxc upstream**: a payload-backed
    ``core.Tensor`` cannot be exported zero-copy (``__dlpack__`` raises;
    ``evoxc.from_dlpack`` is host-memory v1 only), and assembling chunk results
    on the device would need a second compiled graph per call.
    :func:`_iter_plan_chunks` is the seam a future runner uses once that lands
    (see its docstring); the SR loss -- the hot path -- already avoids the round
    trip entirely (:func:`sr_fitness_dense`).

    Raises:
        DenseUnsupported: the population cannot be handled densely -- callers
            fall back to ``evoxc_engine.decode_population`` + ``evaluate``.
    """
    # The wired call sites hand us only ``pop_size``, so the ``auto`` rows rule
    # (``EVOGP_DENSE_AUTO_BUDGET_MB`` / ``EVOGP_DENSE_AUTO_MAX_ROWS``) is enforced
    # here, where BOTH halves of the envelope's "shape" are known: the data-row
    # count from ``x`` and the population's ``(pop_size, tree_len)`` from the
    # population tensors.  A decline is reported the way every other "do not use
    # dense" answer is, i.e. as DenseUnsupported, and the caller falls back to the
    # specialized engine.  ``per_row=True`` sees one row per tree and is never
    # gated.  The check runs BEFORE the host conversion of ``x`` (the population
    # shape is read from ``.shape`` only), so a declined call pays neither the D2H
    # nor (on the memoized path) anything else.
    pop_size, tree_len = _population_shape(batch_subtree_size)
    rows = None if per_row else _decline_rows(x)
    if rows is not None:
        decline = auto_rows_decline(rows, pop_size=pop_size, tree_len=tree_len)
        if decline:
            raise DenseUnsupported(decline)
    query = _resident_array(x, np.float32, "x")
    if query.ndim != 2:
        raise ValueError(
            f"x must be 2-D (rows, {int(input_len)}), got shape {query.shape}"
        )
    if rows is None and not per_row:  # row count known only after the conversion
        decline = auto_rows_decline(
            int(query.shape[0]), pop_size=pop_size, tree_len=tree_len
        )
        if decline:
            raise DenseUnsupported(decline)
    # -- the ``auto`` trace-cap raise (the cap-axis twin of the rows rule) -----
    # Hot path: the row stride comes from ``.shape``, so a population that fits
    # the cap changes nothing here -- no host conversion, no device sync.
    # The pinned/explicit cases short-circuit here for the same reason: under a
    # pinned cap (``max_tree_len=`` / a non-empty ``EVOGP_DENSE_MAX_TREE_LEN``)
    # or a non-``auto`` ``EVOGP_EVAL_MODE`` :func:`_cap_decision` can only return
    # the base cap, so skipping the block costs no host read and is bit-identical
    # to the previous behaviour.  The raise itself ROUNDS the realized longest tree
    # UP to a power of two (:func:`_next_power_of_two`): ``L`` is part of the graph
    # key, so an exact raise recompiled every time ``longest`` drifted -- i.e. once
    # per generation of a GP loop -- while a power-of-two band shares one graph.
    cap = _resolve_max_tree_len(max_tree_len)
    if (
        pop_size
        and tree_len is not None
        and tree_len > cap
        and not trace_cap_pinned(max_tree_len)
        and resolve_eval_mode() == "auto"
    ):
        longest = _population_longest(batch_subtree_size)
        plan_rows = (
            None
            if per_row
            else (rows if rows is not None else int(query.shape[0]))
        )
        decline = auto_cap_decline(
            longest,
            pop_size=pop_size,
            rows=plan_rows,
            max_tree_len=max_tree_len,
        )
        if decline:
            raise DenseUnsupported(decline, device_fallback=True)
        raised = auto_trace_cap(
            longest,
            tree_len,
            pop_size=pop_size,
            rows=plan_rows,
            max_tree_len=max_tree_len,
        )
        if raised != cap:
            max_tree_len = raised
    entry = _plan_entry(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        int(query.shape[0]),
        per_row=per_row,
        variant=VALUE_VARIANT,
        use_mse=True,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
        iree_device_index=iree_device_index,
    )
    plan = entry.plan
    if query.shape[1] != plan.input_len:
        raise ValueError(
            f"x has {query.shape[1]} columns but the population expects "
            f"input_len={plan.input_len}"
        )
    if per_row and query.shape[0] != plan.pop_size:
        raise ValueError(
            f"per_row=True needs x.shape[0] == {plan.pop_size} (pop_size), got "
            f"{query.shape[0]}"
        )
    with entry.lock:
        rows = run_plan(
            plan,
            entry.executable,
            query,
            backend_name=entry.backend_name,
            device=entry.device,
            staged=_entry_payloads(entry, x),
        )
    if per_row:
        rows = np.ascontiguousarray(rows[:, :1])
    else:
        rows = rows[:, :, None]
    # ONE conversion, no copy: ``run_plan`` hands back a fresh contiguous
    # float32 array, so ``from_numpy`` wraps it as the result (fresh, writable,
    # requires_grad=False) and the only upload is the caller-facing device copy.
    tensor = torch.from_numpy(rows).requires_grad_(False)
    target = out_device if out_device is not None else getattr(x, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


def sr_fitness_dense(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    inputs: Any,
    labels: Any,
    use_MSE: bool = True,
    *,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
) -> torch.Tensor:
    """Symbolic-regression fitness via the dense evaluator (IN-GRAPH loss).

    Same contract as :func:`evoxc_engine.sr_fitness`: ``inputs`` is
    ``(N, input_len)``, ``labels`` is ``(N, output_len)`` and the result is a
    ``(pop_size,)`` float32 POSITIVE loss ``sum_N sum_out err / N`` (MSE or MAE)
    on ``inputs``' device unless ``out_device``; NaN/inf propagate unsanitized.

    Host crossings: ``inputs`` and ``labels`` are each converted to host ONCE
    per distinct tensor (:data:`DENSE_INPUT_MEMO` -- a training loop passes the
    same dataset every generation, so the steady state does no D2H at all), the
    population's decode/plan and the ``build_cached`` key derivation are skipped
    for an unchanged population (:func:`_plan_entry`), and the payloads are
    uploaded once and re-used by reference while ``inputs``/``labels`` stay put
    (:func:`_entry_payloads`).  The ONLY engine result is the ``(pop_size,)``
    per-chunk loss sum, assembled in float32 and divided by ``N`` host-side
    (:func:`run_loss_plan`): no prediction crosses back.  ``use_MSE`` is a static
    cfg entry, so MSE and MAE are two compiled graphs per shape signature.

    A call that is going to decline (:func:`auto_rows_decline`) now declines
    from ``inputs.shape`` BEFORE any host conversion, so a declined attempt pays
    nothing at all: the row count and the population's ``(pop_size, tree_len)``
    both come from ``.shape``, which is what the memory envelope
    (``pop_size * tree_len * rows * 4`` bytes) is computed from.
    """
    pop_size, tree_len = _population_shape(batch_subtree_size)
    rows = _decline_rows(inputs)
    if rows is not None:
        decline = auto_rows_decline(rows, pop_size=pop_size, tree_len=tree_len)
        if decline:
            raise DenseUnsupported(decline)
    xt = _resident_input(inputs, int(input_len), "inputs")
    if xt.shape[0] <= 0:
        raise ValueError("sr_fitness needs at least one data point, got 0 rows")
    if rows is None:  # not a torch-tensor-shaped input: gate after the conversion
        decline = auto_rows_decline(
            int(xt.shape[0]), pop_size=pop_size, tree_len=tree_len
        )
        if decline:
            raise DenseUnsupported(decline)
    # -- the ``auto`` trace-cap raise (the cap-axis twin of the rows rule) -----
    # Hot path: the row stride comes from ``.shape``, so a population that fits
    # the cap changes nothing here -- no host conversion, no device sync.
    # The pinned/explicit cases short-circuit here for the same reason: under a
    # pinned cap (``max_tree_len=`` / a non-empty ``EVOGP_DENSE_MAX_TREE_LEN``)
    # or a non-``auto`` ``EVOGP_EVAL_MODE`` :func:`_cap_decision` can only return
    # the base cap, so skipping the block costs no host read and is bit-identical
    # to the previous behaviour.  The raise itself ROUNDS the realized longest tree
    # UP to a power of two (:func:`_next_power_of_two`): ``L`` is part of the graph
    # key, so an exact raise recompiled every time ``longest`` drifted -- i.e. once
    # per generation of a GP loop -- while a power-of-two band shares one graph.
    cap = _resolve_max_tree_len(max_tree_len)
    if (
        pop_size
        and tree_len is not None
        and tree_len > cap
        and not trace_cap_pinned(max_tree_len)
        and resolve_eval_mode() == "auto"
    ):
        longest = _population_longest(batch_subtree_size)
        plan_rows = rows if rows is not None else int(xt.shape[0])
        decline = auto_cap_decline(
            longest,
            pop_size=pop_size,
            rows=plan_rows,
            max_tree_len=max_tree_len,
        )
        if decline:
            raise DenseUnsupported(decline, device_fallback=True)
        raised = auto_trace_cap(
            longest,
            tree_len,
            pop_size=pop_size,
            rows=plan_rows,
            max_tree_len=max_tree_len,
        )
        if raised != cap:
            max_tree_len = raised
    label = _resident_array(labels, np.float32, "labels")
    if label.ndim != 2 or label.shape != (xt.shape[0], int(output_len)):
        raise ValueError(
            f"labels must have shape ({xt.shape[0]}, {int(output_len)}), got "
            f"{label.shape}"
        )
    entry = _plan_entry(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        int(input_len),
        int(output_len),
        int(xt.shape[0]),
        per_row=False,
        variant=LOSS_VARIANT,
        use_mse=bool(use_MSE),
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
        iree_device_index=iree_device_index,
    )
    with entry.lock:
        totals = run_loss_plan(
            entry.plan,
            entry.executable,
            xt,
            label,
            backend_name=entry.backend_name,
            device=entry.device,
            staged=_entry_payloads(entry, inputs, labels),
        )
    # ONE conversion, no copy: ``run_loss_plan`` returns a FRESH float32 array
    # (never a memoized buffer), so the tensor the pipeline mutates in place is
    # this call's own.
    tensor = torch.from_numpy(totals).requires_grad_(False)
    target = out_device if out_device is not None else getattr(inputs, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


# ---------------------------------------------------------------------------
# benchmark-support runner: build once, run many (no readback, no re-key)
# ---------------------------------------------------------------------------
#
# ``build_dense_runner`` / ``DenseRunner`` are the dense half of the two frozen
# benchmark-support runner APIs; the specialized half is
# ``evoxc_engine.build_specialized_runner``.  The shared contract (exact
# signatures, what the build phase costs, what ``.run()`` does per backend and
# what it returns) is documented in ``RUNNER_HOOKS.md`` (this directory) and
# pinned by ``runner_check.py``; the short version:
#
# * the BUILD phase does everything that is not "the compiled forward pass" --
#   host preprocessing (:func:`plan_dense`), the ONE compile of the shape
#   signature (:func:`build_dense`), and the staging of every chunk payload
#   (preparation + upload) through the readback-free seam
#   (:func:`_iter_plan_chunks` with an injected payload memo), plus one warm-up
#   run that does the staging;
# * ``.run()`` executes ONLY the forward pass: one ``evoxc.run`` over the staged
#   payloads, returning the RAW backend result -- device-resident on a device
#   backend (no readback, no host assembly, no torch conversion) and
#   host-backed on numpy (nothing is uploaded or read back there by
#   construction).  N calls are N identical passes;
# * nothing in the library calls them, they do not change the public route
#   behaviour (``EVOGP_EVAL_MODE`` is NOT consulted: the caller picks the route
#   by picking the builder) and they are not part of the training path.

#: The payload keys/staging of a dense runner are plain dict entries of the
#: injected seam memo (``(host_array, device_tensor_or_None)`` per key).


class DenseRunner:
    """A whole population behind ONE compiled dense graph and staged inputs.

    The constructor IS the build phase (both :func:`build_dense_runner` and a
    direct ``DenseRunner(...)`` build it the same way):

    1. duck-type the population (:func:`evoxc_engine._population_inputs`) and
       validate/convert ``inputs`` once (``(N, input_len)`` float32, ``N ==
       pop_size`` when ``per_row``) -- the same semantics as :func:`evaluate_dense`;
    2. :func:`plan_dense` the host preprocessing + chunk shape + graph specs and
       :func:`build_dense` the ONE executable of the shape signature (a shape
       already compiled is a pure cache hit -- ONE entry for the whole
       population, against ``pop_size`` on the specialized route);
    3. stage every payload of the (single) chunk ONCE via
       :func:`_iter_plan_chunks` with an injected memo (:attr:`staged`), which
       also performs the warm-up run that fills it.

    The runner is single-chunk BY CONTRACT: a chunked plan raises
    :class:`DenseUnsupported` at build time, because a chunked result cannot be
    assembled without a host (or a second compiled graph) step, which is outside
    "``.run()`` executes only the forward pass".  Raise ``EVOGP_DENSE_BUDGET_MB``
    to keep a benchmark shape in one chunk.

    Read-only attributes (diagnostics for harnesses):
        plan: the :class:`DensePlan` (``trace_len`` / ``pop_size`` / ``n_rows`` /
            chunk shape / ``peak_bytes`` / variant ...; see :func:`describe_plan`).
        executable: the compiled (or cache-hit) evoxc executable.
        staged: the payload memo (payload key -> ``(host_array,
            device_tensor_or_None)``) every ``.run()`` re-uses.
        backend_name / device: the resolved :func:`_resolve_backend` pair
            (frozen for every ``.run()``); ``device_resident`` is
            ``backend_name != "numpy"``.
        per_row, input_len, output_len, pop_size, n_rows (the graph's own ``N``:
            ``1`` for ``per_row=True``).

    The result stays in the backend's own memory: read it back yourself
    (``raw.to(evoxc.core.Device("cpu", 0)).numpy()`` on a device backend,
    ``raw.numpy()`` on numpy) -- that is the consumer's copy, not a runner step.
    """

    def __init__(
        self,
        population_or_forest: Any,
        inputs: Any,
        *,
        per_row: bool = False,
        backend: Optional[str] = None,
        iree_device_index: Optional[int] = None,
        max_tree_len: Optional[int] = None,
        budget_mb: Optional[float] = None,
        update_mode: Optional[str] = None,
    ) -> None:
        values, types, sizes, input_len, output_len = _population_inputs(
            population_or_forest
        )
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.per_row = bool(per_row)
        xin = _input_to_numpy(inputs, self.input_len, "inputs")
        plan = plan_dense(
            values,
            types,
            sizes,
            self.input_len,
            self.output_len,
            int(xin.shape[0]),
            per_row=self.per_row,
            max_tree_len=max_tree_len,
            budget_mb=budget_mb,
            update_mode=update_mode,
            backend=backend,
        )
        if self.per_row and int(xin.shape[0]) != plan.pop_size:
            raise ValueError(
                f"per_row=True needs inputs.shape[0] == {plan.pop_size} "
                f"(pop_size), got {xin.shape[0]}"
            )
        if plan.p_chunks > 1 or plan.n_chunks > 1:
            raise DenseUnsupported(
                "build_dense_runner needs a single-chunk plan, got "
                f"{plan.p_chunks}x{plan.n_chunks} chunks of "
                f"(P_chunk={plan.p_chunk}, N_chunk={plan.n_chunk}): a chunked "
                "result cannot be assembled without a host (or a second "
                "compiled graph) step, which is outside this API's "
                f"no-readback contract.  Raise {BUDGET_MB_ENV} (currently "
                f"{plan.budget_mb}) to keep this shape in one chunk."
            )
        self.plan = plan
        self.executable = build_dense(
            plan, backend=backend, iree_device_index=iree_device_index
        )
        self.backend_name, self.device, _ = _resolve_backend(backend, iree_device_index)
        self.device_resident = self.backend_name != "numpy"
        self.staged: Dict[Any, Tuple[np.ndarray, Any]] = {}
        self._x = np.ascontiguousarray(xin)
        chunks = list(
            _iter_plan_chunks(
                plan,
                self.executable,
                self._x,
                backend_name=self.backend_name,
                device=self.device,
                device_resident=self.device_resident,
                staged=self.staged,
            )
        )
        if len(chunks) != 1:  # pragma: no cover - guarded by the chunk check above
            raise DenseUnsupported(
                f"the plan yielded {len(chunks)} chunks but the runner contract "
                "is exactly one"
            )

    @property
    def invocations(self) -> int:
        """How many ``evoxc.run`` calls one :meth:`run` issues (always one chunk)."""
        return 1

    def run(self) -> Any:
        """Execute ONE forward pass -- the only thing a harness times.

        Returns:
            The RAW ``evoxc.run`` result of this plan's single chunk: an ``evoxc``
            runtime tensor of shape ``(pop_size, n_rows)`` (``n_rows == 1`` for
            ``per_row=True``), **device-resident** on a device backend
            (``iree``: no readback, no host assembly, no torch conversion) and
            host-backed on the ``numpy`` backend (which never leaves the host,
            so there is nothing to read back and nothing is copied).

        Nothing else happens: the payloads staged by the build phase are re-used
        by reference (no re-preparation, no re-upload), the executable is held
        directly (no ``build_cached`` key derivation, no ``_executable``
        lookup), the plan is never rebuilt and no result is converted.  The
        static ``cfg`` is re-passed by value exactly as the frozen engine does
        (:func:`_iter_plan_chunks`).
        """
        iterator = _iter_plan_chunks(
            self.plan,
            self.executable,
            self._x,
            backend_name=self.backend_name,
            device=self.device,
            device_resident=self.device_resident,
            staged=self.staged,
        )
        _chunk, raw = next(iterator)
        return raw


def build_dense_runner(
    population_or_forest: Any,
    inputs: Any,
    *,
    per_row: bool = False,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    budget_mb: Optional[float] = None,
    update_mode: Optional[str] = None,
) -> DenseRunner:
    """Build a repeatable dense-evaluator runner (the frozen benchmark API).

    Args:
        population_or_forest: any ``Forest``-like container (duck-typed on
            ``batch_node_value`` / ``batch_node_type`` / ``batch_subtree_size``
            / ``input_len`` / ``output_len``; no import of the container).
        inputs: ``x`` with the semantics of :func:`evaluate_dense` -- ``(N,
            input_len)``, converted to float32 host once for the build; ``N ==
            pop_size`` when ``per_row=True`` (tree ``i`` sees row ``i``), else
            the data points every tree is evaluated on.
        per_row: per-tree inputs (default ``False``: every tree sees all rows).
        backend / iree_device_index / max_tree_len / budget_mb / update_mode:
            exactly :func:`evaluate_dense`'s knobs (backend and device index
            override the environment; the trace cap and the chunk budget are the
            dense-specific ones).

    Returns:
        A :class:`DenseRunner`; ``.run()`` is the compiled forward pass only (one
        ``evoxc.run`` over the staged payloads; see its docstring for the exact
        return contract).

    Raises:
        DenseUnsupported: the population is outside the dense subset
            (multi-output, over the trace cap, malformed) OR the plan chunks --
            see the class docstring.
        TypeError: the container is not Forest-like (or the population tensors
            are not 2-D).
        ValueError: bad ``inputs`` shape, a ``per_row`` row-count mismatch, or an
            unsupported ``max_tree_len`` / ``budget_mb`` / ``update_mode``.
        ValueError: an unsupported backend name (from :func:`_resolve_backend`).

    Note:
        The ``EVOGP_EVAL_MODE`` policy is NOT consulted: this API selects the
        dense route explicitly, exactly like ``eval_mode="dense"`` does, but
        without the silent fallback -- an unsupported population raises here
        instead of running on the specialized engine.
    """
    return DenseRunner(
        population_or_forest,
        inputs,
        per_row=per_row,
        backend=backend,
        iree_device_index=iree_device_index,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
    )
