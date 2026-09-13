"""Triton fused-kernel evaluator for EvoGP's single-output symbolic-regression case.

Round r6 prototype (``test/benchmark/r6/``).  It replaces the *engine body* of one
``problem.evaluate(forest)`` call with ONE fused Triton kernel launch and therefore
tests the r5 RCA hypothesis (``../r5/RCA.md`` §8/§9): the ETL dense route spends
≈4.2-4.3 ms of a ~5.4-5.8 ms call in per-call host glue (10 payload uploads, dense
decode, host loss assembly, readback) plus a 310-device-op dispatch-bound graph,
while the legacy CUDA engine does the whole population in 3 device ops
(``memset`` + fused tree-walk kernel + averaging kernel) with zero readback.

What this module is
-------------------
* **Staging** (declared): ``staging = "dense_engine.prepare_dense"``.  The population
  is decoded/validated by :func:`evogp.tree.dense_engine.prepare_dense` (the exact
  same host-side, vectorized staging the dense ETL route feeds its graph with:
  ``(L, P)`` columns ``[opcode, func_id, const_value, var_index, child1, child2,
  child3]`` plus ``tree_len``), so the two routes see byte-identical staging.  The
  columns are uploaded to the device ONCE, at build time.  Nothing is decoded in
  the kernel from the raw ``(pop, max_tree_len)`` schema.
* **Runtime interpreter** (legacy-style): one program per tree, data rows as block
  lanes, a *reverse prefix walk* over node positions ``tree_len-1 .. 0`` (children
  always have a larger index than their parent), the node-level opcode/function-id
  dispatch done with **scalar** conditions (never per-lane ``if``: a vector
  condition aborts the Triton compiler process — see ``CONTEXT.md``), and a
  per-tree value scratch ``(pid, position, lane)`` that is written once per node and
  read by its parent.  The legacy kernel's ``MAX_STACK = 1024`` operand stack is not
  needed: positions are read through the validated child columns instead of through
  a scalar stack pointer, so no stack-depth limit exists here.
* **Loss in the kernel**: ``sum_N (label - pred)^2 / N`` (MSE, the default) or
  ``sum_N |label - pred| / N`` (MAE) is reduced inside the same single kernel and
  stored straight to the device output.  **Zero per-call uploads, zero per-call
  readback, one launch per call**; the result stays device-resident, exactly like the
  legacy ``tree_SR_fitness`` kernel's output.
* **Build once / run many**: :func:`build_evaluator` returns a
  :class:`TritonEvaluator` whose :meth:`~TritonEvaluator.run` is one launch over
  resident staging.  :func:`sr_fitness_triton` is a drop-in replacement for
  :func:`evogp.tree.dense_engine.sr_fitness_dense` (identical signature and
  positive-loss contract) with a process-level cache keyed on the population's
  tensor identity, so the r6 type-1 driver can patch the dense seam and keep the
  unmodified public API (``problem.evaluate(forest)``).

Op coverage (declared, complete for the engine's 29-id table)
------------------------------------------------------------
Every function id of ``ENGINE_SPEC.md`` §3 is implemented, in the arity its node
*base type* implies (the legacy dispatch is keyed on the base type and the id is an
if/else chain inside it):

* ``VAR`` / ``CONST`` leaves;
* ``BFUNC`` ids 1-13: ``+``, ``-``, ``*``, ``/``, ``loose_div``, ``pow``,
  ``loose_pow``, ``max``, ``min``, ``<``, ``>``, ``<=``, ``>=``;
* ``UFUNC`` ids 14-28: ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``, ``tanh``,
  ``log``, ``loose_log``, ``exp``, ``inv``, ``loose_inv``, ``neg``, ``abs``,
  ``sqrt``, ``loose_sqrt``;
* ``TFUNC`` (and any unmapped base type, which the legacy kernel also routes to the
  ternary branch): id ignored, always ``var1 > 0 ? var2 : var3``.

A type/id combination that matches no branch leaves the value at ``0.0f`` — the
legacy kernel's and the oracle's own rule, reproduced here (never a wrong number).
``/`` is the ENGINE_SPEC select (``var2 == 0 ? NaN : var1 / var2``) evaluated with
``div_rn`` (correctly rounded IEEE float32 division), so it agrees with the
correctly-rounded oracle bit-for-bit whenever the operands are finite; ``+ - *`` use
plain float32 add/sub/mul and are bit-exact.  The libdevice transcendentals
(``sin``…``sqrt``) are correctly *rounded to the platform's own libdevice values*,
which can differ from the correctly-rounded oracle in the last ulp; the required
fixture ops are ``VAR``/``CONST``/``+ - * /`` only.

Contract
--------
``TritonEvaluator.run()`` returns, for the default ``output="loss"``,

* ``(pop_size,)`` **float32**, **positive** loss ``sum_N sum_out err / N``
  (never divided by ``output_len``), fresh/contiguous/``requires_grad=False``, on
  ``inputs``' device (or ``out_device``).  NaN/±inf propagate unsanitized — exactly
  :func:`evogp.tree.dense_engine.sr_fitness_dense`'s contract, which the caller
  (``Forest.SR_fitness`` → ``SymbolicRegression.evaluate``) negates;

and for ``output="values"`` the per-tree prediction ``(pop_size, N)`` float32 (used
by ``validate_triton.py`` for per-op exactness against the oracle).  ``loss`` is the
production path; ``values`` exists for validation only.

Usage (from the repository root, project venv, CUDA-visible GPU)::

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 python test/benchmark/r6/run_triton_eval64.py \
        --fixture test/goldens/forest_random.npz --reps 25 --tag triton_f64 --json-out /tmp/o.json
"""
from __future__ import annotations

import os
import statistics
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

try:  # triton >= 3.0 ships the portable shim; the cuda module is the fallback
    from triton.language.extra import libdevice as _libdevice
except ImportError:  # pragma: no cover - depends on the installed triton wheel
    from triton.language.extra.cuda import libdevice as _libdevice

__all__ = [
    "STAGING",
    "TritonUnsupported",
    "TritonEvaluator",
    "build_evaluator",
    "sr_fitness_triton",
    "install_dense_seam",
    "uninstall_dense_seam",
    "reset_stats",
    "stats",
    "covered_ops",
    "coverage_report",
    "OP_NAMES",
]

# --------------------------------------------------------------------------- #
# constants — duplicated from ENGINE_SPEC.md / src/evogp/tests/reference_eval.py
# --------------------------------------------------------------------------- #
DELTA = np.float32(1e-9)  # defs.h DELTA = 1E-9f (float32!)
MAX_VAL = float(np.float32(1e9))  # defs.h MAX_VAL = 1E9f
NAN = float("nan")

#: Declared staging: the dense route's own host-side staging (see module docstring).
STAGING = "dense_engine.prepare_dense"
SEAM_TARGET = "evogp.tree.dense_engine.sr_fitness_dense"

# opcode column values (dense_engine): 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC/unknown
OP_VAR, OP_CONST, OP_UFUNC, OP_BFUNC, OP_TFUNC = 0, 1, 2, 3, 4
OP_INACTIVE = 5  # padding / tree-length tail: never visited by the walk

#: ids implemented per base type (the kernel's own dispatch tables).
BFUNC_IDS = tuple(range(1, 14))  # + - * / loose_div pow loose_pow max min < > <= >=
UFUNC_IDS = tuple(range(14, 29))  # sin cos tan sinh cosh tanh log loose_log exp inv ...
TFUNC_ID = 0

OP_NAMES: Dict[Tuple[int, int], str] = {(OP_VAR, 0): "VAR", (OP_CONST, 0): "CONST"}
OP_NAMES.update({(OP_BFUNC, i): n for i, n in zip(BFUNC_IDS, (
    "+", "-", "*", "/", "loose_div", "pow", "loose_pow", "max", "min", "<", ">", "<=", ">="))})
OP_NAMES.update({(OP_UFUNC, i): n for i, n in zip(UFUNC_IDS, (
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "log", "loose_log", "exp", "inv",
    "loose_inv", "neg", "abs", "sqrt", "loose_sqrt"))})
OP_NAMES[(OP_TFUNC, TFUNC_ID)] = "if"

#: Ops whose float32 result is required to be bit-exact against the oracle.
EXACT_OPS = ("VAR", "CONST", "+", "-", "*", "/")

_LIBDEVICE_UNAVAILABLE = tuple(
    name for name in ("div_rn", "sin", "cos", "tan", "sinh", "cosh", "tanh", "log",
                      "exp", "sqrt", "pow", "copysign")
    if not hasattr(_libdevice, name)
)
if _LIBDEVICE_UNAVAILABLE:  # a triton build without these cannot run this kernel
    raise ImportError(
        f"triton libdevice is missing {_LIBDEVICE_UNAVAILABLE}; the r6 evaluator needs them"
    )


class TritonUnsupported(RuntimeError):
    """The population cannot be handled by the Triton evaluator (never silently).

    Raised instead of :class:`evogp.tree.dense_engine.DenseUnsupported` on purpose:
    ``Forest.SR_fitness`` catches *only* ``DenseUnsupported`` and would silently
    re-run the call on the specialized engine (the round's known hazard), so this
    exception is a distinct type and propagates.
    """


# --------------------------------------------------------------------------- #
# the fused kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _sr_eval_kernel(
    op_ptr,        # (L, P) int32   opcode: 0 VAR 1 CONST 2 UFUNC 3 BFUNC 4 TFUNC
    fid_ptr,       # (L, P) int32   function id (clamped to [0, 29] by the staging)
    cval_ptr,      # (L, P) f32     constant value (CONST nodes)
    vidx_ptr,      # (L, P) int32   variable index (VAR nodes)
    c1_ptr,        # (L, P) int32   first child position  (prefix order)
    c2_ptr,        # (L, P) int32   second child position
    c3_ptr,        # (L, P) int32   third child position
    tlen_ptr,      # (P,)   int32   tree length = subtree_size[0]
    vbuf_ptr,      # (P, L, BLOCK_N) f32 value scratch (one row per node position)
    x_ptr,         # (N, INPUT_LEN) f32 data
    y_ptr,         # (N,)   f32 labels (single output)
    out_ptr,       # (P,) loss   or  (P, N) values
    P, L, N, INPUT_LEN, N_F32,
    BLOCK_N: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    USE_MSE: tl.constexpr,
    OUT_VALUES: tl.constexpr,
):
    """One program per tree; lanes are data rows; reverse prefix walk over nodes.

    Every node-level condition is a SCALAR (uniform across the program): a per-lane
    ``if`` aborts the Triton compiler process (round r6 known trap).
    """
    pid = tl.program_id(0)
    tree_len = tl.load(tlen_ptr + pid)
    lanes = tl.arange(0, BLOCK_N)
    row = pid * L * BLOCK_N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    nan = float("nan")

    for rb in tl.static_range(N_BLOCKS):
        offs = rb * BLOCK_N + lanes
        rmask = offs < N
        # ---- reverse prefix walk: children (larger index) before their parent ----
        for i in range(tree_len - 1, -1, -1):
            # the staged columns are (P, L) row-major -> node i of tree pid is at pid*L + i
            idx = pid * L + i
            op = tl.load(op_ptr + idx)
            if op == 0:  # VAR leaf: push x[row, var_index]
                vi = tl.load(vidx_ptr + idx)
                v = tl.load(x_ptr + offs * INPUT_LEN + vi, mask=rmask, other=0.0)
            elif op == 1:  # CONST leaf
                v = tl.zeros([BLOCK_N], dtype=tl.float32) + tl.load(cval_ptr + idx)
            elif op == 2:  # unary function (arity 1 from the base type)
                a = tl.load(
                    vbuf_ptr + row + tl.load(c1_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                fid = tl.load(fid_ptr + idx)
                if fid == 14:
                    v = _libdevice.sin(a)
                elif fid == 15:
                    v = _libdevice.cos(a)
                elif fid == 16:
                    v = _libdevice.tan(a)
                elif fid == 17:
                    v = _libdevice.sinh(a)
                elif fid == 18:
                    v = _libdevice.cosh(a)
                elif fid == 19:
                    v = _libdevice.tanh(a)
                elif fid == 20:  # log: no zero special case (log(0) = -inf)
                    v = _libdevice.log(a)
                elif fid == 21:  # loose_log
                    v = tl.where(a == 0.0, -1e9, _libdevice.log(tl.abs(a)))
                elif fid == 22:
                    v = _libdevice.exp(a)
                elif fid == 23:  # inv: 0 -> NaN, else 1/a (correctly rounded)
                    v = tl.where(
                        a == 0.0, nan,
                        _libdevice.div_rn(tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0, a),
                    )
                elif fid == 24:  # loose_inv: |a| <= DELTA -> copy_sign(DELTA, a)
                    b = tl.where(
                        tl.abs(a) <= 1e-9, _libdevice.copysign(1e-9, a), a
                    )
                    v = _libdevice.div_rn(tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0, b)
                elif fid == 25:
                    v = -a
                elif fid == 26:
                    v = tl.abs(a)
                elif fid == 27:  # sqrt: negative -> NaN
                    v = _libdevice.sqrt(a)
                elif fid == 28:  # loose_sqrt: a <= 0 -> sqrt(|a|)
                    v = _libdevice.sqrt(tl.where(a <= 0.0, tl.abs(a), a))
                else:  # matches no branch in the legacy chain -> 0.0f
                    v = tl.zeros([BLOCK_N], dtype=tl.float32)
            elif op == 3:  # binary function (arity 2)
                a = tl.load(
                    vbuf_ptr + row + tl.load(c1_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                b = tl.load(
                    vbuf_ptr + row + tl.load(c2_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                fid = tl.load(fid_ptr + idx)
                if fid == 1:
                    v = a + b
                elif fid == 2:
                    v = a - b
                elif fid == 3:
                    v = a * b
                elif fid == 4:  # /: var2 == 0 -> NaN (select, never a trap)
                    v = tl.where(b == 0.0, nan, _libdevice.div_rn(a, b))
                elif fid == 5:  # loose_div
                    b2 = tl.where(tl.abs(b) <= 1e-9, _libdevice.copysign(1e-9, b), b)
                    v = _libdevice.div_rn(a, b2)
                elif fid == 6:
                    v = _libdevice.pow(a, b)
                elif fid == 7:  # loose_pow
                    v = tl.where(
                        (a == 0.0) & (b == 0.0), 0.0,
                        _libdevice.pow(tl.abs(a), b),
                    )
                elif fid == 8:  # max is a ternary, NOT fmaxf: max(v, NaN) = NaN
                    v = tl.where(a >= b, a, b)
                elif fid == 9:  # min: min(NaN, v) = v
                    v = tl.where(a <= b, a, b)
                elif fid == 10:
                    v = tl.where(a < b, 1.0, -1.0) + tl.zeros([BLOCK_N], dtype=tl.float32)
                elif fid == 11:
                    v = tl.where(a > b, 1.0, -1.0) + tl.zeros([BLOCK_N], dtype=tl.float32)
                elif fid == 12:
                    v = tl.where(a <= b, 1.0, -1.0) + tl.zeros([BLOCK_N], dtype=tl.float32)
                elif fid == 13:
                    v = tl.where(a >= b, 1.0, -1.0) + tl.zeros([BLOCK_N], dtype=tl.float32)
                else:  # no branch (e.g. a BFUNC node carrying id 0 or > 13) -> 0.0f
                    v = tl.zeros([BLOCK_N], dtype=tl.float32)
            else:  # TFUNC (and any unmapped base type): id ignored, always `if`
                a = tl.load(
                    vbuf_ptr + row + tl.load(c1_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                b = tl.load(
                    vbuf_ptr + row + tl.load(c2_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                c = tl.load(
                    vbuf_ptr + row + tl.load(c3_ptr + idx) * BLOCK_N + lanes,
                    mask=rmask, other=0.0,
                )
                v = tl.where(a > 0.0, b, c)
            tl.store(vbuf_ptr + row + i * BLOCK_N + lanes, v, mask=rmask)
        # ---- root value -> per-lane error / output -----------------------------
        root = tl.load(vbuf_ptr + row + lanes, mask=rmask, other=0.0)
        if OUT_VALUES:
            tl.store(out_ptr + pid * N + offs, root, mask=rmask)
        else:
            y = tl.load(y_ptr + offs, mask=rmask, other=0.0)
            diff = y - root
            if USE_MSE:
                err = diff * diff
            else:
                err = tl.abs(diff)
            # ``select`` (not a multiply): a masked-off lane may hold NaN/inf from
            # the ``other=0.0`` fills of a partial block and must not poison the sum
            acc += tl.where(rmask, err, 0.0)
    if not OUT_VALUES:
        total = tl.sum(acc, axis=0)
        tl.store(out_ptr + pid, _libdevice.div_rn(total, N_F32))


def _coverage_ok(op: int, fid: int) -> bool:
    """Is this staged ``(opcode, function id)`` pair a real dispatch branch?

    ``False`` = the legacy kernel's if/else chain has no matching branch, so the
    node value is ``0.0f``; the kernel reproduces exactly that (never a wrong
    number, but worth reporting).
    """
    if op in (OP_VAR, OP_CONST):
        return True
    if op == OP_UFUNC:
        return int(fid) in UFUNC_IDS
    if op == OP_BFUNC:
        return int(fid) in BFUNC_IDS
    return int(fid) == TFUNC_ID  # TFUNC ignores its id


def covered_ops() -> List[str]:
    """The implemented dispatch table, as ``["<kind>:<name>", ...]``."""
    names = ["VAR", "CONST"]
    names += [f"BFUNC:{OP_NAMES[(OP_BFUNC, i)]}" for i in BFUNC_IDS]
    names += [f"UFUNC:{OP_NAMES[(OP_UFUNC, i)]}" for i in UFUNC_IDS]
    names += ["TFUNC:if"]
    return names


def coverage_report(prep) -> Dict[str, Any]:
    """Per-population op coverage: used pairs, implemented pairs, fall-throughs."""
    used: Dict[str, int] = {}
    fallthrough: Dict[str, int] = {}
    for i in range(prep.trace_len):
        real = prep.real[i]
        for p in np.nonzero(real)[0]:
            op = int(prep.opcode[i, p])
            fid = int(prep.func_id[i, p])
            key = OP_NAMES.get((op, fid))
            if key is None:  # no branch at all: kind/id mismatch, value 0.0f
                fallthrough[f"op{op}:fid{fid}"] = fallthrough.get(f"op{op}:fid{fid}", 0) + 1
            else:
                used[key] = used.get(key, 0) + 1
    return {
        "used_ops": sorted(used),
        "used_ops_histogram": dict(sorted(used.items())),
        "fallthrough_nodes": dict(sorted(fallthrough.items())),
        "uncovered_ops": [],
    }


def _next_pow2(n: int) -> int:
    value = 1
    while value < int(n):
        value *= 2
    return max(1, value)


# --------------------------------------------------------------------------- #
# build / run
# --------------------------------------------------------------------------- #
class TritonEvaluator:
    """One compiled fused kernel + device-resident staging (build once, run many).

    ``run()`` is ONE kernel launch with no upload and no readback.  The staging is
    uploaded at build time; ``ensure_inputs`` re-uploads ``x``/``labels`` only when
    their tensor identity changed (so a fixed benchmark population never uploads
    inside a timed call, while a training loop that swaps the data tensor still
    gets correct numbers).
    """

    def __init__(
        self,
        prep,
        batch_node_value: torch.Tensor,
        batch_node_type: torch.Tensor,
        batch_subtree_size: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        use_MSE: bool = True,
        output: str = "loss",
        block_n: Optional[int] = None,
        max_tree_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if output not in ("loss", "values"):
            raise ValueError(f"output must be 'loss' or 'values', got {output!r}")
        self.prep = prep
        self.output = output
        self.use_MSE = bool(use_MSE)
        self.max_tree_len = max_tree_len
        self.P = int(prep.pop_size)
        self.L = int(prep.trace_len)
        self.input_len = int(prep.input_len)
        self.output_len = int(prep.output_len)
        if self.output_len != 1:
            raise TritonUnsupported(
                f"output_len={self.output_len}: the Triton evaluator implements the "
                "single-output SR protocol only"
            )
        self.device = torch.device(device if device is not None else inputs.device)
        self.forest_key = (
            _tensor_key(batch_node_value),
            _tensor_key(batch_node_type),
            _tensor_key(batch_subtree_size),
        )
        self.coverage = coverage_report(prep)
        # ---- staging upload (once) -------------------------------------------
        t0 = time.perf_counter()
        self._op = self._to_device(prep.opcode.T, torch.int32)      # (P, L)
        self._fid = self._to_device(prep.func_id.T, torch.int32)
        self._cval = self._to_device(prep.const_value.T, torch.float32)
        self._vidx = self._to_device(prep.var_index.T, torch.int32)
        self._c1 = self._to_device(prep.child1.T, torch.int32)
        self._c2 = self._to_device(prep.child2.T, torch.int32)
        self._c3 = self._to_device(prep.child3.T, torch.int32)
        self._tlen = self._to_device(prep.tree_len.astype(np.int32), torch.int32)
        self._N: Optional[int] = None
        self._block_n: Optional[int] = None
        self._n_blocks: Optional[int] = None
        self._forced_block_n = int(block_n) if block_n else None
        self._vbuf: Optional[torch.Tensor] = None
        self._data_key: Optional[Tuple[Any, ...]] = None
        self._x: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self.stage_upload_ms = (time.perf_counter() - t0) * 1000.0
        self.build_ms: Optional[float] = None
        self.n_runs = 0
        self.n_builds = 1
        self.ensure_inputs(inputs, labels)
        # ---- compile: one throwaway launch that triggers the JIT ---------------
        t1 = time.perf_counter()
        scratch = self._alloc_out(self.N)
        self._launch(scratch, count=False)  # the compile probe is not an "invocation"
        torch.cuda.synchronize(self.device)
        self.build_ms = (time.perf_counter() - t1) * 1000.0
        self.construct_total_ms = self.stage_upload_ms + self.build_ms

    # -- helpers -------------------------------------------------------------
    def _to_device(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=self.device)

    def _alloc_out(self, n_rows: int) -> torch.Tensor:
        if self.output == "loss":
            return torch.empty((self.P,), dtype=torch.float32, device=self.device)
        return torch.empty((self.P, n_rows), dtype=torch.float32, device=self.device)

    def ensure_inputs(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Stage ``x``/``labels`` (re-upload only when their identity changed)."""
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TritonUnsupported("inputs and labels must be torch tensors")
        if int(inputs.shape[1]) != self.input_len:
            raise TritonUnsupported(
                f"inputs has {int(inputs.shape[1])} columns but the population expects "
                f"input_len={self.input_len}"
            )
        if tuple(labels.shape) != (int(inputs.shape[0]), 1):
            raise TritonUnsupported(
                f"labels must have shape ({int(inputs.shape[0])}, 1) for the single-output "
                f"protocol, got {tuple(labels.shape)}"
            )
        key = (_tensor_key(inputs), _tensor_key(labels))
        if key == self._data_key:
            return
        n = int(inputs.shape[0])
        if n != self._N or self._vbuf is None:
            self._N = n
            self._block_n = max(_next_pow2(n), self._forced_block_n or 0)
            self._n_blocks = -(-n // self._block_n)
            self._vbuf = torch.empty(
                (self.P, self.L, self._block_n), dtype=torch.float32, device=self.device
            )
        x = inputs.detach()
        y = labels.detach()
        if x.device != self.device or x.dtype != torch.float32 or not x.is_contiguous():
            x = x.to(device=self.device, dtype=torch.float32).contiguous()
        if y.device != self.device or y.dtype != torch.float32 or not y.is_contiguous():
            y = y.to(device=self.device, dtype=torch.float32).contiguous()
        self._x = x
        self._y = y.reshape(-1) if y.dim() > 1 else y
        self._data_key = key
        _STATS["data_uploads"] += 1

    @property
    def N(self) -> int:
        if self._N is None:
            raise TritonUnsupported("inputs were never staged")
        return int(self._N)

    def _launch(self, out: torch.Tensor, count: bool = True) -> None:
        """The ONE kernel launch.  ``count=False`` for the build-time compile probe."""
        _sr_eval_kernel[(self.P,)](
            self._op, self._fid, self._cval, self._vidx,
            self._c1, self._c2, self._c3, self._tlen,
            self._vbuf, self._x, self._y, out,
            self.P, self.L, self.N, self.input_len, float(self.N),
            BLOCK_N=self._block_n,
            N_BLOCKS=self._n_blocks,
            USE_MSE=self.use_MSE,
            OUT_VALUES=(self.output == "values"),
            num_warps=min(8, max(1, self._block_n // 32)),
        )
        if count:
            self.n_runs += 1
            _STATS["triton_runs"] += 1

    # -- public --------------------------------------------------------------
    def run(self) -> torch.Tensor:
        """ONE launch: ``(pop_size,)`` positive loss (default) or ``(pop_size, N)``."""
        out = self._alloc_out(self.N)
        self._launch(out)
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "staging": STAGING,
            "output": self.output,
            "use_MSE": self.use_MSE,
            "pop": self.P,
            "trace_len": self.L,
            "n_rows": self.N,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "block_n": self._block_n,
            "launches_per_call": 1,
            "kernel": "sr_eval_kernel",
            "num_warps": min(8, max(1, (self._block_n or 1) // 32)),
            "op_coverage": covered_ops(),
            "used_ops": self.coverage["used_ops"],
            "fallthrough_nodes": self.coverage["fallthrough_nodes"],
            "uncovered_ops": self.coverage["uncovered_ops"],
            "stage_upload_ms": self.stage_upload_ms,
            "compile_ms": self.build_ms,
            "construct_total_ms": self.construct_total_ms,
            "n_runs": self.n_runs,
        }


def _tensor_key(t: torch.Tensor) -> Tuple[int, Tuple[int, ...], str, int]:
    """Identity of a tensor for the build cache: (data_ptr, shape, dtype, version).
    ``_version`` is bumped by torch on every in-place write, so an in-place-mutated
    population (the training loop's ``forest.batch_node_value.copy_`` pattern)
    misses the cache and rebuilds instead of returning stale numbers.
    """
    return (int(t.data_ptr()), tuple(int(s) for s in t.shape), str(t.dtype), int(t._version))


def _build_key(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    n_rows: int,
    use_MSE: bool,
    output: str,
    block_n: int,
    max_tree_len: Optional[int],
    device: str,
) -> Tuple[Any, ...]:
    """The evaluator cache key -- population/data identity + shape + mode (decode-free)."""
    return (
        _tensor_key(batch_node_value),
        _tensor_key(batch_node_type),
        _tensor_key(batch_subtree_size),
        int(input_len),
        int(output_len),
        int(n_rows),
        bool(use_MSE),
        output,
        int(block_n),
        max_tree_len,
        device,
    )


# --------------------------------------------------------------------------- #
# build + process-level cache + counters
# --------------------------------------------------------------------------- #
#: built evaluators, keyed on population identity + data shape + mode (small LRU)
_EVAL_CACHE: "OrderedDict[Tuple[Any, ...], TritonEvaluator]" = OrderedDict()
_CACHE_LIMIT = 8
_LAST_BUILT: Optional[TritonEvaluator] = None
_STATS: Dict[str, Any] = {
    "seam_calls": 0,
    "triton_runs": 0,
    "builds": 0,
    "cache_hits": 0,
    "data_uploads": 0,
    "fallbacks": 0,
    "patched": False,
}


def reset_stats() -> None:
    for key in _STATS:
        _STATS[key] = False if key == "patched" else 0
    return None


def stats() -> Dict[str, Any]:
    """A copy of the process-level counters (seam calls, runs, builds, uploads)."""
    return dict(_STATS)


def last_built_evaluator() -> Optional[TritonEvaluator]:
    """The most recently *constructed* evaluator (``None`` if none was built yet).

    The r6 type-1 envelope builds the evaluator lazily inside the first
    ``problem.evaluate`` call, so this is the only handle on its compile/staging
    timings (``describe()``) from the driver.
    """
    return _LAST_BUILT


def _resolve_tensors(
    forest: Any,
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: Optional[int],
    output_len: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if forest is not None:
        for name in ("batch_node_value", "batch_node_type", "batch_subtree_size"):
            if not hasattr(forest, name):
                raise TritonUnsupported(f"forest has no {name!r} attribute")
        batch_node_value = forest.batch_node_value
        batch_node_type = forest.batch_node_type
        batch_subtree_size = forest.batch_subtree_size
        input_len = int(forest.input_len if input_len is None else input_len)
        output_len = int(forest.output_len if output_len is None else output_len)
    if batch_node_value is None or batch_node_type is None or batch_subtree_size is None:
        raise TritonUnsupported("a population (forest= or the three node tensors) is required")
    if input_len is None or output_len is None:
        raise TritonUnsupported("input_len and output_len are required with raw node tensors")
    return (
        batch_node_value, batch_node_type, batch_subtree_size, int(input_len), int(output_len)
    )


def build_evaluator(
    forest: Any = None,
    inputs: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    *,
    batch_node_value: Any = None,
    batch_node_type: Any = None,
    batch_subtree_size: Any = None,
    input_len: Optional[int] = None,
    output_len: Optional[int] = None,
    use_MSE: bool = True,
    output: str = "loss",
    block_n: Optional[int] = None,
    max_tree_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    cache: bool = True,
) -> TritonEvaluator:
    """Build (or fetch from the cache) the fused evaluator for one population.

    ``forest`` may be an :class:`evogp.tree.Forest` (duck-typed on
    ``batch_node_value``/``batch_node_type``/``batch_subtree_size``/``input_len``/
    ``output_len``) or the three raw node tensors may be passed by keyword together
    with ``input_len``/``output_len``.  ``inputs`` is ``(N, input_len)`` and
    ``labels`` ``(N, 1)``; both must be CUDA float32 (anything else is copied).

    Raises:
        TritonUnsupported: the population cannot be handled (multi-output, a
            malformed structure, a tree longer than ``EVOGP_DENSE_MAX_TREE_LEN``,
            a non-finite/out-of-range variable index or function id -- everything
            :func:`~evogp.tree.dense_engine.prepare_dense` rejects).  This is a
            *different* type from ``DenseUnsupported`` on purpose (see the class).
    """
    from evogp.tree import dense_engine  # local import: keeps the module importable alone

    (
        batch_node_value, batch_node_type, batch_subtree_size, input_len, output_len
    ) = _resolve_tensors(
        forest, batch_node_value, batch_node_type, batch_subtree_size, input_len, output_len
    )
    if inputs is None or labels is None:
        raise TritonUnsupported("inputs and labels are required")
    # The cache key is built from TENSOR IDENTITIES ONLY -- never from the decode:
    # `prepare_dense` is the dense route's host decode (~0.7 ms for 64x63, ~1.4 ms
    # for 200x64 on this box, r5 `results/q1_decomp/`) and a hot path that pays it
    # per call would just re-implement the glue this prototype exists to remove.
    key = _build_key(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        int(inputs.shape[0]),
        bool(use_MSE),
        output,
        int(block_n or 0),
        max_tree_len,
        str(device if device is not None else inputs.device),
    )
    if cache and key in _EVAL_CACHE:
        _STATS["cache_hits"] += 1
        evaluator = _EVAL_CACHE[key]
        evaluator.ensure_inputs(inputs, labels)
        return evaluator
    try:
        prep = dense_engine.prepare_dense(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            input_len,
            output_len,
            max_tree_len=max_tree_len,
        )
    except dense_engine.DenseUnsupported as exc:
        raise TritonUnsupported(f"triton staging rejected the population: {exc}") from exc
    evaluator = TritonEvaluator(
        prep,
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        inputs,
        labels,
        use_MSE=use_MSE,
        output=output,
        block_n=block_n,
        max_tree_len=max_tree_len,
        device=device,
    )
    _STATS["builds"] += 1
    global _LAST_BUILT
    _LAST_BUILT = evaluator
    if cache:
        _EVAL_CACHE[key] = evaluator
        while len(_EVAL_CACHE) > _CACHE_LIMIT:
            _EVAL_CACHE.popitem(last=False)
    return evaluator


def sr_fitness_triton(
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
    """Drop-in replacement for :func:`evogp.tree.dense_engine.sr_fitness_dense`.

    Signature and contract are identical (positive ``(pop_size,)`` float32 loss on
    ``inputs``' device, MSE by default, NaN/inf unsanitized); the ETL-only kwargs
    (``backend``, ``iree_device_index``, ``budget_mb``, ``update_mode``) are accepted
    and ignored -- the Triton path has no ETL backend to choose.  ``max_tree_len``
    is honored (it is the staging trace cap).

    Per call: cache lookup (population identity + data identity) → at most one
    re-upload of ``x``/``labels`` if their tensors changed → ONE kernel launch.
    """
    _STATS["seam_calls"] += 1
    if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TritonUnsupported("inputs and labels must be torch tensors")
    evaluator = build_evaluator(
        None,
        inputs,
        labels,
        batch_node_value=batch_node_value,
        batch_node_type=batch_node_type,
        batch_subtree_size=batch_subtree_size,
        input_len=int(input_len),
        output_len=int(output_len),
        use_MSE=bool(use_MSE),
        output="loss",
        max_tree_len=max_tree_len,
    )
    evaluator.ensure_inputs(inputs, labels)  # no-op unless the data tensors changed
    out = evaluator.run()
    if out_device is not None and out.device != torch.device(out_device):
        out = out.to(out_device)
    return out


# --------------------------------------------------------------------------- #
# the dense seam patch (type-1 envelope)
# --------------------------------------------------------------------------- #
_PREVIOUS_SEAM: Optional[Any] = None


def install_dense_seam(reset: bool = True) -> None:
    """Point ``evogp.tree.dense_engine.sr_fitness_dense`` at :func:`sr_fitness_triton`.

    ``Forest.SR_fitness`` resolves the seam through the module attribute at call
    time (``from . import dense_engine`` + ``dense_engine.should_use_dense`` /
    ``dense_engine.sr_fitness_dense``), so an attribute rebind is enough -- ``src/``
    stays untouched.  ``EVOGP_EVAL_MODE=dense`` must be set BEFORE importing evogp:
    under the default ``auto`` policy on the numpy backend ``should_use_dense``
    returns False and the call never reaches the seam (it would silently use the
    specialized engine instead).
    """
    global _PREVIOUS_SEAM
    from evogp.tree import dense_engine

    if _PREVIOUS_SEAM is None:
        _PREVIOUS_SEAM = dense_engine.sr_fitness_dense
    dense_engine.sr_fitness_dense = sr_fitness_triton
    _STATS["patched"] = True
    if reset:
        reset_stats()
        _STATS["patched"] = True


def uninstall_dense_seam() -> None:
    """Restore the original :func:`evogp.tree.dense_engine.sr_fitness_dense`."""
    global _PREVIOUS_SEAM
    from evogp.tree import dense_engine

    if _PREVIOUS_SEAM is not None:
        dense_engine.sr_fitness_dense = _PREVIOUS_SEAM
        _PREVIOUS_SEAM = None
    _STATS["patched"] = False


def seam_is_patched() -> bool:
    """Is ``evogp.tree.dense_engine.sr_fitness_dense`` currently :func:`sr_fitness_triton`?"""
    from evogp.tree import dense_engine

    return dense_engine.sr_fitness_dense is sr_fitness_triton


def binding_style() -> str:
    """How the seam is bound (reported verbatim in the r6 records)."""
    return (
        "module-attribute rebind of evogp.tree.dense_engine.sr_fitness_dense "
        "(resolved at call time by Forest.SR_fitness via the module attribute)"
    )
