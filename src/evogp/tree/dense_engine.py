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
One master switch, read **per call** (exactly like ``EVOGP_EVOXC_BACKEND`` in
``evoxc_engine._resolve_backend``), case-insensitive, default ``auto``:

``auto`` (default)
    the measured policy, implemented once in :func:`resolve_route` (the rows half
    in :func:`auto_rows_decline`), which also decides the BACKEND per call: with
    no explicit backend (neither ``EVOGP_EVOXC_BACKEND`` nor ``backend=``) and a
    GPU usable from torch, ``auto`` runs the DENSE route on ``iree`` from
    ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP`` (default
    :data:`DEFAULT_AUTO_MIN_POP` = 64) up and the SPECIALIZED route on ``numpy``
    below it -- and on numpy for every dense decline/fallback
    (:class:`DenseUnsupported` included), because a per-tree iree compile
    (271-284 ms/tree) can never amortize.  On a box with no device, and with an
    explicit backend, the historical rule is unchanged: dense iff the resolved
    backend is not ``numpy`` *and* ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP``
    *and*, on ``per_row=False`` calls,
    ``rows <= EVOGP_DENSE_AUTO_MAX_ROWS`` (default
    :data:`DEFAULT_AUTO_MAX_ROWS` = 2048).  The wired call sites pass ``pop_size``
    only, so the rows half of the rule is enforced inside
    :func:`evaluate_dense`, where the data-row count is known
    (:func:`sr_fitness_dense` goes through it too); ``per_row=True`` (one row per
    tree) and an explicit ``EVOGP_EVAL_MODE=dense`` are never gated, and a
    decline surfaces as the usual :class:`DenseUnsupported` fallback.
    Rationale: the dense path pays ONE compile for the whole population (3.1-3.4 s
    at ``L=32``, 6.1 s at the default cap ``L=64``) against 271-284 ms *per
    distinct structure* on iree, so the warm per-call win (0.5 ms at ``P=16``,
    ~7 ms at ``P=64``, ~34 ms at ``P=256``, ~141 ms at ``P=1000``) pays that
    compile back after ~6300/460/95/23 calls -- hence the ``pop_size`` gate at the
    population where a GP run is a few hundred generations in (lower
    ``EVOGP_DENSE_AUTO_MIN_POP`` to 16 for a very long run), while warm dense is
    1.3-3.2x faster at small ``N`` where both sides are measured directly and
    1.18x *slower* at ``P=16/N=10000`` -- its cost grows with the data-row count
    while the specialized engine's is rows-flat (1.24-1.94 ms/tree from N=64 to
    N=10000) -- hence the rows ceiling, ~2x below the tightest measured crossover
    (~3.4-6.8k rows at ``P=16``).
``dense``
    always attempt the dense path (still falling back on
    :class:`DenseUnsupported`); with no explicit backend it uses the device
    backend on a device box (a user forcing dense wants the fast dense route) and
    ``numpy`` elsewhere.
``specialized``
    never use dense; the specialized engine only, on the explicitly named (or
    engine-default) backend.

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
   expensive one.
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
``EVOGP_DENSE_AUTO_MAX_ROWS``  2048        rows ceiling of the ``auto``
                                            policy on ``per_row=False`` calls;
                                            a decline -> ``DenseUnsupported``
                                            fallback; ``per_row=True`` and an
                                            explicit ``dense`` are never gated
``EVOGP_DENSE_MAX_TREE_LEN``    32          max trace length ``L``; longer trees
                                            -> :class:`DenseUnsupported`
``EVOGP_DENSE_BUDGET_MB``       2048        chunking memory budget (MiB)
``EVOGP_DENSE_UPDATE_MODE``     grow        ``grow`` / ``concat`` / ``scatter``
``EVOGP_DENSE_ALLOW_UNSAFE_SCATTER``        unset  permits ``scatter`` on iree
=============================== =========== ==================================
Every knob is read per call (see the module's configuration section), so a
process can change strategy between calls.  Backend/device/opt-level knobs, the
shared compile cache (:data:`DENSE_CACHE` is the very same
:data:`evoxc_engine.CACHE`) and the ``EVOXC_DISABLE_LOCATIONS=1`` tracing speedup
are inherited unchanged from :mod:`evogp.tree.evoxc_engine`.

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
   term) and, on ``per_row=False`` calls,
   ``rows <= EVOGP_DENSE_AUTO_MAX_ROWS`` (2048, enforced inside
   :func:`evaluate_dense`; ``per_row=True`` and an explicit ``dense`` are never
   gated; a decline is the usual :class:`DenseUnsupported` fallback), and it never
   selects the numpy backend for the dense route on its own.
   Firm-up crossovers: ~3.4-6.8k rows warm at ``P=16`` (dense 1.18x slower at
   ``P=16/N=10000``).  The cheapest large win identified: pass the population's
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
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc

from .utils import NType
from .evoxc_engine import (  # private but deliberately reused: parity by construction
    BACKEND_ENV,
    CACHE,
    _CPU,
    _as_numpy,
    _emit_binary,
    _emit_ternary,
    _emit_unary,
    _env_int,
    _operand,
    _population_inputs,
    _quiet,
    _resolve_backend,
    _trace_const,
)

__all__ = [
    "ALLOW_UNSAFE_SCATTER_ENV",
    "AUTO_MAX_ROWS_ENV",
    "AUTO_MIN_POP_ENV",
    "BUDGET_MB_ENV",
    "DEFAULT_AUTO_MAX_ROWS",
    "DEFAULT_AUTO_MIN_POP",
    "DEFAULT_BUDGET_MB",
    "DEFAULT_EVAL_MODE",
    "DEFAULT_MAX_TREE_LEN",
    "DEFAULT_PRUNE",
    "DENSE_CACHE",
    "DensePlan",
    "DensePrep",
    "DenseRunner",
    "DenseUnsupported",
    "EVAL_MODES",
    "EVAL_MODE_ENV",
    "FULL_FUNC_IDS",
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
    "auto_rows_decline",
    "build_dense",
    "build_dense_runner",
    "dense_peak_bytes",
    "describe_plan",
    "device_available",
    "evaluate_dense",
    "plan_dense",
    "prepare_dense",
    "resolve_auto_max_rows",
    "resolve_auto_min_pop",
    "resolve_backend_name",
    "resolve_eval_mode",
    "resolve_prune",
    "resolve_route",
    "should_use_dense",
    "sr_fitness_dense",
]

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: Master switch (``EVOGP_EVAL_MODE``): ``dense`` / ``specialized`` / ``auto``.
#: The ONE policy knob; :func:`resolve_eval_mode` / :func:`resolve_route` read it
#: per call, exactly like ``evoxc_engine._resolve_backend`` reads the backend
#: variable.
EVAL_MODE_ENV = "EVOGP_EVAL_MODE"
#: Population-size threshold of the ``auto`` policy
#: (``EVOGP_DENSE_AUTO_MIN_POP``): dense+iree from here up on a device box,
#: specialized+numpy below it -- see :func:`resolve_route`.
AUTO_MIN_POP_ENV = "EVOGP_DENSE_AUTO_MIN_POP"
#: Data-row ceiling of the ``auto`` policy (``EVOGP_DENSE_AUTO_MAX_ROWS``) for
#: ``per_row=False`` calls: dense cost grows
#: with ``rows`` while the specialized cost is rows-flat, so above the crossover
#: dense loses at any population -- see :func:`auto_rows_decline`.
AUTO_MAX_ROWS_ENV = "EVOGP_DENSE_AUTO_MAX_ROWS"
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

EVAL_MODES = ("dense", "specialized", "auto")
DEFAULT_EVAL_MODE = "auto"
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

#: Pop-size threshold of the ``auto`` policy, measured on the current pruned
#: dense chain (one dense compile at ``L=32`` on iree is 3.1-3.4 s / 6.1 s at
#: ``L=64``): the warm per-call win of dense+iree over specialized+numpy is
#: 0.5 ms at ``P=16``, ~7 ms at ``P=64``, ~34 ms at ``P=256`` and ~141 ms at
#: ``P=1000``, i.e. the compile is paid back after ~6300 / ~460 / ~95 / ~23
#: calls.  64 is the population where a GP run is a few hundred generations in;
#: lower it to 16 (``EVOGP_DENSE_AUTO_MIN_POP=16``) for a very long run.
DEFAULT_AUTO_MIN_POP = 64
#: Rows crossover measured on iree (RTX A6000, ``L=32``) for the smallest
#: population the ``auto`` policy admits: dense is 1.49x / 1.30x faster than the
#: specialized engine at P=16 with 64 / 1024 rows and 1.18x *slower* at 10000
#: rows (same-process A/B), i.e. the crossover is ~3.4-4.9k rows there (7.5k at
#: P=32, ~11k at P=64).  The default is the measured SAFE boundary, ~2x below the
#: tightest crossover; env-overridable.
DEFAULT_AUTO_MAX_ROWS = 2048

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
    """

    def __init__(self, reason: str):
        super().__init__(f"dense evaluation unsupported: {reason}")
        self.reason = reason


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


def resolve_auto_max_rows(value: Optional[int] = None) -> int:
    """Resolve the ``auto`` policy's row-count ceiling (``EVOGP_DENSE_AUTO_MAX_ROWS``)."""
    maximum = (
        _env_int(AUTO_MAX_ROWS_ENV, DEFAULT_AUTO_MAX_ROWS)
        if value is None
        else int(value)
    )
    if maximum < 1:
        raise ValueError(
            f"{AUTO_MAX_ROWS_ENV}={maximum} must be >= 1 (it is a data-row ceiling)"
        )
    return maximum


def _rows_decline_reason(
    rows: Optional[int], *, mode: str, auto_max_rows: Optional[int]
) -> Optional[str]:
    """The rows rule itself: why ``auto`` declines dense at ``rows`` (else ``None``).

    Private single source of the half-rule; :func:`auto_rows_decline` (public, mode
    resolved from ``eval_mode``) and :func:`resolve_route` (which already has the
    resolved mode) both go through it, so the threshold exists once.
    """
    if rows is None or mode != "auto":
        return None
    maximum = resolve_auto_max_rows(auto_max_rows)
    if int(rows) <= maximum:
        return None
    return (
        f"auto policy: {int(rows)} data rows exceed {AUTO_MAX_ROWS_ENV}={maximum}; "
        "the dense graph is rows-linear while the specialized engine is rows-flat "
        "(dense measured 1.18x slower at P=16/N=10000 on iree, see "
        "dense_bench_results.json / dense_bench_findings.md). "
        "Set EVOGP_EVAL_MODE=dense to force it."
    )


def auto_rows_decline(
    rows: Optional[int],
    *,
    eval_mode: Optional[str] = None,
    auto_max_rows: Optional[int] = None,
) -> Optional[str]:
    """Why the ``auto`` policy declines dense at ``rows`` data rows (else ``None``).

    The single source of the rows rule: dense cost grows with the data-row count
    while the specialized engine's warm cost is rows-flat, so ``auto`` keeps dense
    only up to :data:`DEFAULT_AUTO_MAX_ROWS`.  ``rows=None`` (unknown, e.g. a
    ``per_row=True`` call) never declines.  An explicit ``EVOGP_EVAL_MODE=dense``
    always wins -- the user's opt-in is not second-guessed, the same way an
    explicit ``specialized`` is never overridden.
    """
    return _rows_decline_reason(
        rows, mode=resolve_eval_mode(eval_mode), auto_max_rows=auto_max_rows
    )


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


@dataclass(frozen=True)
class Route:
    """The route policy's decision for one call.

    ``backend`` is always a concrete backend NAME (``"numpy"`` / ``"iree"``) that
    the call site passes explicitly to :func:`evaluate_dense` /
    :func:`sr_fitness_dense` (when ``use_dense``) or to
    :func:`evoxc_engine.evaluate` / :func:`evoxc_engine.sr_fitness` (``backend``,
    or ``fallback_backend`` after a :class:`DenseUnsupported`), so the route -- not
    an ambient environment variable -- decides where a call runs.
    """

    use_dense: bool
    backend: str
    fallback_backend: str
    reason: str = ""


def resolve_route(
    pop_size: int,
    *,
    rows: Optional[int] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    eval_mode: Optional[str] = None,
    auto_min_pop: Optional[int] = None,
    auto_max_rows: Optional[int] = None,
) -> Route:
    """The ONE route policy: dense or specialized, and on which backend.

    Every wired call site (:meth:`evogp.tree.Forest.forward` /
    ``Forest.batch_forward`` / ``Forest.SR_fitness``, :meth:`evogp.tree.Tree.forward`
    / ``Tree.SR_fitness``) asks this, so the policy exists exactly once;
    :func:`should_use_dense` is the boolean view of the same decision.
    ``pop_size`` is the population about to be evaluated (``1`` for a ``Tree``).

    Precedence: an explicitly provided backend (the ``backend=`` argument or a
    non-empty ``EVOGP_EVOXC_BACKEND``) first, then ``EVOGP_EVAL_MODE``, then the
    device probe.

    * ``specialized`` -> never dense, on the explicitly named backend or the
      engine's own default (``numpy``).
    * ``dense`` -> always attempt dense; with no explicit backend it uses the
      device backend on a device box (a user forcing dense wants the fast dense
      route) and ``numpy`` elsewhere.
    * ``auto`` with an explicit backend -> the historical rule: dense iff that
      backend is not ``numpy`` *and* ``pop_size >= EVOGP_DENSE_AUTO_MIN_POP``
      *and* ``rows`` (when given) is inside ``EVOGP_DENSE_AUTO_MAX_ROWS``.
    * ``auto`` without an explicit backend -> the policy owns the backend: dense on
      ``iree`` from ``EVOGP_DENSE_AUTO_MIN_POP`` (64) up when a device is
      available, and ``numpy`` for the specialized route -- below the threshold,
      with no device, and after a :class:`DenseUnsupported` (a per-tree iree
      compile costs 271-284 ms/tree and can never amortize).
    """
    mode = resolve_eval_mode(eval_mode)
    explicit = str(backend).strip() if backend else os.environ.get(BACKEND_ENV, "").strip()

    if mode == "specialized":
        name = resolve_backend_name(backend, iree_device_index)
        return Route(False, name, name, "EVOGP_EVAL_MODE=specialized: specialized engine only")

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
    decline = _rows_decline_reason(rows, mode=mode, auto_max_rows=auto_max_rows)
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
    # No explicit backend: the policy picks the backend too.
    if not device_available():
        return Route(
            False,
            "numpy",
            "numpy",
            "auto policy: no CUDA/HIP device usable from torch -> specialized on numpy",
        )
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


def should_use_dense(
    pop_size: int,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    eval_mode: Optional[str] = None,
    auto_min_pop: Optional[int] = None,
    rows: Optional[int] = None,
    auto_max_rows: Optional[int] = None,
) -> bool:
    """The ONE policy decision: try the dense evaluator for this population?

    The boolean view of :func:`resolve_route` (it delegates, so the rule exists
    exactly once): ``True`` means the call site attempts the dense evaluator with
    ``resolve_route(...).backend`` and, on :class:`DenseUnsupported`, continues on
    the specialized engine with ``resolve_route(...).fallback_backend``.
    ``pop_size`` is the population about to be evaluated (``1`` for a ``Tree``).
    """
    return resolve_route(
        pop_size,
        rows=rows,
        backend=backend,
        iree_device_index=iree_device_index,
        eval_mode=eval_mode,
        auto_min_pop=auto_min_pop,
        auto_max_rows=auto_max_rows,
    ).use_dense


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
    multi-output, a structural doubt, the ``auto`` rows ceiling -- and numpy is
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
    """
    raw = type_t.astype(np.int32)
    base = raw & int(NType.TYPE_MASK)
    rows = np.arange(L, dtype=np.int32)[:, None]
    real = rows < tree_len[None, :]
    if ((raw & int(NType.OUT_NODE)) != 0)[real].any():
        raise DenseUnsupported(
            "an OUT_NODE-tagged node in a single-output tree (the specialized "
            "decoder raises ValueError here)"
        )
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
    """
    backend_name, device, options = _resolve_backend(backend, iree_device_index)
    fn = _dense_loss_step if plan.variant == LOSS_VARIANT else _dense_step
    if backend_name == "numpy":
        return evoxc.build_cached(
            fn,
            *plan.specs,
            backend=backend_name,
            device=device,
            cache=DENSE_CACHE,
        )
    return evoxc.build_cached(
        fn,
        *plan.specs,
        backend=backend_name,
        device=device,
        cache=DENSE_CACHE,
        **options,
    )


# ---------------------------------------------------------------------------
# runtime: chunked execution
# ---------------------------------------------------------------------------


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
) -> np.ndarray:
    """Run every chunk of ``plan`` and assemble the ``(P, N)`` result (host).

    The result is the transposed / per-tree layout of
    :func:`evoxc_engine.evaluate`'s return value: ``(P, 1)`` rows for
    ``per_row=True``, ``(P, N)`` otherwise.
    """
    prep = plan.prep
    out = np.empty((prep.pop_size, plan.n_rows), dtype=np.float32)
    for (p0, p_chunk, n0, n_chunk), raw in _iter_plan_chunks(
        plan, executable, x, backend_name=backend_name, device=device
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


def run_loss_plan(    plan: DensePlan,
    executable: Any,
    x: np.ndarray,
    labels: np.ndarray,
    *,
    backend_name: str,
    device: Any,
) -> np.ndarray:
    """Run the loss variant and assemble the ``(pop_size,)`` loss (host).

    Only ``(p_chunk,)``-scale chunk sums ever cross back to the host; they are
    accumulated in float32 and divided by ``N`` (the data-point count) once, so
    the result is ``sum_N sum_out err / N`` -- the same formula, and the same
    float32 "sum then divide" order, as :func:`evoxc_engine.sr_fitness` (the
    in-graph reduction order differs, ~1e-7 relative on the fixtures).
    """
    if plan.variant != LOSS_VARIANT:
        raise ValueError(
            f"run_loss_plan needs a {LOSS_VARIANT!r} plan, got {plan.variant!r}"
        )
    totals = np.zeros((plan.prep.pop_size,), dtype=np.float32)
    for (p0, p_chunk, _n0, _n_chunk), raw in _iter_plan_chunks(
        plan, executable, x, labels, backend_name=backend_name, device=device
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

    Host crossings: ``x`` is converted to host ONCE and the assembled prediction
    is created as a torch tensor ONCE; the chunk results come back through
    :func:`run_plan` (one readback per chunk -- exactly one for the default
    unchunked plan).  A fully device-resident VALUE path (no readback, no H2D at
    all) is **blocked on evoxc upstream**: a payload-backed ``core.Tensor`` cannot
    be exported zero-copy (``__dlpack__`` raises; ``evoxc.from_dlpack`` is
    host-memory v1 only), and assembling chunk results on the device would need a
    second compiled graph per call.  :func:`_iter_plan_chunks` is the seam a
    future runner uses once that lands (see its docstring); the SR loss -- the
    hot path -- already avoids the round trip entirely
    (:func:`sr_fitness_dense`).

    Raises:
        DenseUnsupported: the population cannot be handled densely -- callers
            fall back to ``evoxc_engine.decode_population`` + ``evaluate``.
    """
    query = _as_numpy(x, np.float32, "x")
    if query.ndim != 2:
        raise ValueError(
            f"x must be 2-D (rows, {int(input_len)}), got shape {query.shape}"
        )
    # The wired call sites hand us only ``pop_size``, so the ``auto`` rows rule
    # (``EVOGP_DENSE_AUTO_MAX_ROWS``) is enforced here, where the row count is
    # known: a decline is reported the way every other "do not use dense" answer
    # is, i.e. as DenseUnsupported, and the caller falls back to the specialized
    # engine.  ``per_row=True`` sees one row per tree and is never gated.
    decline = auto_rows_decline(None if per_row else int(query.shape[0]))
    if decline:
        raise DenseUnsupported(decline)
    plan = plan_dense(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        int(query.shape[0]),
        per_row=per_row,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
    )
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
    executable = build_dense(
        plan, backend=backend, iree_device_index=iree_device_index
    )
    backend_name, device, _ = _resolve_backend(backend, iree_device_index)
    rows = run_plan(plan, executable, query, backend_name=backend_name, device=device)
    if per_row:
        rows = np.ascontiguousarray(rows[:, :1])
    else:
        rows = rows[:, :, None]
    tensor = torch.from_numpy(rows).contiguous().detach().requires_grad_(False)
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
    (never twice -- this path does not go through :func:`evaluate_dense`), both
    are uploaded once per data chunk, and the ONLY engine result is the
    ``(pop_size,)`` per-chunk loss sum, assembled in float32 and divided by ``N``
    host-side (:func:`run_loss_plan`): no prediction crosses back.  ``use_MSE``
    is a static cfg entry, so MSE and MAE are two compiled graphs per shape
    signature.
    """
    xt = _input_to_numpy(inputs, int(input_len), "inputs")
    if xt.shape[0] <= 0:
        raise ValueError("sr_fitness needs at least one data point, got 0 rows")
    label = _as_numpy(labels, np.float32, "labels")
    if label.ndim != 2 or label.shape != (xt.shape[0], int(output_len)):
        raise ValueError(
            f"labels must have shape ({xt.shape[0]}, {int(output_len)}), got "
            f"{label.shape}"
        )
    # The wired call sites hand us only ``pop_size``, so the ``auto`` rows rule is
    # enforced here (same decline -> DenseUnsupported as evaluate_dense).
    decline = auto_rows_decline(int(xt.shape[0]))
    if decline:
        raise DenseUnsupported(decline)
    plan = plan_dense(
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
    )
    executable = build_dense(
        plan, backend=backend, iree_device_index=iree_device_index
    )
    backend_name, device, _ = _resolve_backend(backend, iree_device_index)
    totals = run_loss_plan(
        plan, executable, xt, label, backend_name=backend_name, device=device
    )
    tensor = torch.from_numpy(totals).contiguous().detach().requires_grad_(False)
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
