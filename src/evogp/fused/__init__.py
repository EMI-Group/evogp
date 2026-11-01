"""The FUSED evaluation route: ONE Triton kernel launch for a whole population.

This package is EvoGP's optional device-native evaluator.  It takes a population
the way the GP stack already holds it -- three prefix-order tensors
(``node_value`` float32, ``node_type`` int16, ``subtree_size`` int16, all
``(pop_size, max_tree_len)``) on a torch device -- and evaluates it with ONE
JIT-compiled Triton kernel launch per (tree, row block) pair: no compiled graph,
no host decode, no host-side assembly and no readback on the value path.  The
SR loss is reduced IN the kernel, so its only device output is a
``(pop_size, n_row_blocks)`` partial chart that a caller sums and divides.

Nothing has to be built.  The route needs ``torch`` and ``triton``, declared as
the optional ``fused`` extra (``pip install evogp[fused]``), and Triton JITs the
kernels at the FIRST call -- a one-time cost of ~1.6 s for the value kernel
(~2.9 s for all three surfaces) from an empty cache, ~2 ms from a warm disk one,
and nothing per call or per generation afterwards.  ``TRITON_CACHE_DIR`` must
point at a writable directory on a box whose ``$HOME`` is read-only.

Two callers use it, both on that legacy triple:

* :mod:`evogp.gp` -- the torch-free functional GP stack, whose
  ``fused_eval.FusedDeviceEvaluator`` hands the kernel its own device buffer
  through ``out=``;
* ``evogp.fused.check`` -- the in-library battery (parity, declines, timing).

The route is a hard, explicit boundary: ``import evogp.gp`` never imports this
package (the functional stack stays torch-free and runs on numpy alone), and a
call this package cannot serve raises :class:`FusedUnsupported` so the caller
decides what to do.

============================== ==================================================
:class:`FusedUnsupported`      per-call decline (the route cannot serve this call)
:func:`fused_available`        ``triton`` importable AND a torch CUDA/HIP device
:func:`envelope_decline`       the memory envelope's verdict for a shape (no raise)
:func:`evaluate_fused`         value path, ``(pop, output_len)`` / ``(pop, N, output_len)``
:func:`sr_fitness_fused`       SR loss path, ``(pop,)`` positive loss
:func:`fused_cache_stats`      this route's own counters (all prefixed ``fused_``)
:func:`clear_fused_cache`      reset them (this route's ONE reset entry)
:func:`resolve_block_rows` /   the ``EVOGP_FUSED_*`` geometry knobs
:func:`resolve_num_warps` /
:func:`resolve_max_scratch_mb`
:func:`resolve_dispatch`       the launch path (``bound`` / ``jit``)
:data:`MAX_OUT_LEN`            the widest multi-output tree the route serves (8)
:mod:`evogp.fused.kernels`     the Triton kernels (the device half)
============================== ==================================================
"""

from __future__ import annotations

from . import kernels
from .engine import (
    MAX_OUT_LEN,
    FusedUnsupported,
    clear_fused_cache,
    envelope_decline,
    evaluate_fused,
    fused_available,
    fused_cache_stats,
    resolve_block_rows,
    resolve_dispatch,
    resolve_max_scratch_mb,
    resolve_num_warps,
    sr_fitness_fused,
)

__all__ = [
    "FusedUnsupported",
    "MAX_OUT_LEN",
    "clear_fused_cache",
    "envelope_decline",
    "evaluate_fused",
    "fused_available",
    "fused_cache_stats",
    "kernels",
    "resolve_block_rows",
    "resolve_dispatch",
    "resolve_max_scratch_mb",
    "resolve_num_warps",
    "sr_fitness_fused",
]
