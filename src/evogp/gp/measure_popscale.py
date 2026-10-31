"""Measurement harness: PER-CALL evaluation of ONE population, population-scaled.

This is COMMITTED EVIDENCE, not a test and not a library entry point: nothing runs
at import time, everything lives inside :func:`main` or the function a lane
dispatch calls.  The torch/triton/``evogp.tree`` boundary is resolved through
:func:`importlib.import_module` inside the functions (the boundary
:mod:`evogp.gp.fused_eval` uses), because the sibling purity test forbids the
literal statement in this subtree.

Where :mod:`evogp.gp.measure_generation` measures a whole GENERATION (evaluation
+ select/crossover/mutate), this harness measures the PRIMARY metric of the
population-scale round: one evaluation call of ONE fixed population, on the same
trees, in every lane.

The population is the shared axis
---------------------------------
At each population size the harness builds ONE ``evogp.gp`` population (the numpy
backend, the round's seed and config) and derives every layout from it: the
``GPState`` itself, its legacy triple (:func:`evogp.gp.encoding.to_legacy_triple`)
and the device channels uploaded from those numpy rows (uploaded once, and
verified bit-identical here, outside every timed block).  So no lane can win by
evaluating a different set of trees.

The lanes
---------
================== ==========================================================
``numpy``          :class:`evogp.gp.problem.SymbolicRegression` on the host --
                   ``evaluate`` (the maximized fitness, the reference of the
                   fitness parity group) and ``predict`` (the ``(pop, rows)``
                   matrix, the reference of the value parity group).
``gp``             :class:`evogp.gp.graph_eval.GraphEvaluator` on ``iree``/device:
                   device channels in, an ``evoxc`` DEVICE fitness tensor out,
                   NO readback inside the timed block.
``gp-fused``       :class:`evogp.gp.fused_eval.FusedDeviceEvaluator`: the Triton
                   kernel reading the device channels through zero-copy DLPack
                   views and writing the fitness buffer through ``out=``.  The
                   record carries ``route_counts`` / ``last_route`` /
                   ``interop_available()`` so it PROVES the device route served
                   the calls instead of falling back to the host hand-off.
``torch-batched``  the plain-torch WHOLE-POPULATION evaluator (one vectorized
                   pass over the population, no python loop over trees), copied
                   from the benchmark campaign's ``BatchedEvaluator`` idiom.
``torch-per-tree`` the plain-torch per-tree python walk (``Tree.forward`` per
                   tree, stacked); scoped to pop <= :data:`SIDE_MAX_POP`.
``forest-auto``    the torch stack's own ``Forest.batch_forward`` and
                   ``Forest.SR_fitness`` under the SHIPPED DEFAULT route
                   (``EVOGP_EVAL_MODE`` unset, so ``auto`` -> fused here); the
                   route the resolver actually picked is recorded.
``forest-fused``   the same two surfaces with ``EVOGP_EVAL_MODE=fused`` forced:
                   the CONTROL, measured in its OWN child process (the route is
                   memoized on first use, so the variable has to be set before
                   ``evogp.tree`` is imported) and only at the widest population.
================== ==========================================================

Two parity groups, never mixed
------------------------------
A maximized FITNESS (``evaluate``, ``SR_fitness`` negated) is never compared
against a raw VALUE (``batch_forward``, ``predict``):

* **fitness parity** -- ``gp``, ``gp-fused``, ``forest-auto``/``forest-fused``
  ``sr_fitness`` against the ``numpy`` lane's ``evaluate``;
* **value parity** -- ``torch-batched``, ``torch-per-tree`` and ``forest-auto`` /
  ``forest-fused`` ``batch_forward`` against the ``numpy`` lane's ``predict``.

Both are reported by :func:`evogp.gp.measure_generation._parity` (``max_rel``,
``max_rel_scaled``, ``finiteness_mismatch``), over the whole population (the value
group over the whole ``(pop, rows)`` matrix).

Protocol
--------
* **One process per population.**  The parent re-executes this module once per
  ``--pop`` (``--pop``/``--lanes`` child mode), sequentially, so a compile, an
  allocator or a leftover buffer can never be charged to another population.
* **Warm medians.**  ``--warmup`` calls are discarded, then ``--reps``
  repetitions are timed and the MEDIAN is reported (min/max/mean/samples go into
  the JSON -- a shared card is noisy).
* **A device sync around every timed repetition** (``torch.cuda.synchronize()``,
  device-wide, so it also waits for the ``evoxc``/IREE work).
* **Cold** is the FIRST call of a lane in the process, one sample, synced: it is
  what a one-shot caller pays (graph compile, Triton JIT).
* **Idle-device guard** before each child (``rocm-smi`` use and pids).
* **Scope-outs** are stated, never silent: a lane above its population ceiling is
  printed and recorded (``scoped_out`` in the JSON, its own report section).
  ``torch-per-tree`` is limited to pop <= :data:`SIDE_MAX_POP` (a python loop over
  trees is already ~0.7 s per call at pop 8192).

Run the grid (one command, the parent spawns one child per population)::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \
        /tmp/rocvenv/bin/python -m evogp.gp.measure_popscale \
            --pops 64,128,256,512,1024,2048,4096,8192 \
            --json src/evogp/gp/evidence/popscale_eval.json

``--pops`` defaults to the eight populations of the round's column, ``--lanes``
to every lane above, ``--rows``/``--reps``/``--warmup``/``--seed`` to
1024/15/5/0.  Nothing is written unless ``--json`` is passed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import encoding, problem
from .encoding import init_population
from .graph_eval import GraphEvaluator, readback_fitness
from .measure_generation import (
    GP_KNOBS,
    POPSCALE_CELLS,
    REPS,
    ROWS,
    SEED,
    WARMUP,
    WORKLOAD,
    _device_kind,
    _evoxc,
    _git_head,
    _median,
    _parity,
    _sync,
    _torch,
    _write,
    guard,
    make_configs,
    versions,
)
__all__ = [
    "POPS",
    "LANES",
    "LANE_SURFACES",
    "SIDE_MAX_POP",
    "measure_pop",
    "report",
    "main",
]

#: The populations of the campaign: the round's column, at one fixed shape.
POPS: Tuple[int, ...] = tuple(int(entry) for entry in POPSCALE_CELLS)

#: The lanes, in report order (``forest-fused`` is the separate-process control).
LANES: Tuple[str, ...] = (
    "numpy",
    "gp",
    "gp-fused",
    "torch-batched",
    "torch-per-tree",
    "forest-auto",
    "forest-fused",
)

#: Per-lane population ceiling of this round; a lane above it is ``scoped_out``.
#: The per-tree python walk is already ~0.7 s per call at pop 8192, which buys
#: nothing at a still-visible pop 1024 (the whole population, one tree at a time).
SIDE_MAX_POP: Dict[str, int] = {"torch-per-tree": 1024}

#: The surfaces of each lane and which quantity each one returns: ``fitness`` is
#: MAXIMIZED (losses negated) and ``value`` is a raw prediction.  ``negate`` marks
#: a surface that returns the POSITIVE loss the caller negates (``SR_fitness``).
LANE_SURFACES: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "numpy": (
        {"surface": "evaluate", "kind": "fitness"},
        {"surface": "predict", "kind": "value"},
    ),
    "gp": ({"surface": "evaluate", "kind": "fitness"},),
    "gp-fused": ({"surface": "evaluate", "kind": "fitness"},),
    "torch-batched": ({"surface": "batch_forward", "kind": "value"},),
    "torch-per-tree": ({"surface": "forward", "kind": "value"},),
    "forest-auto": (
        {"surface": "batch_forward", "kind": "value"},
        {"surface": "sr_fitness", "kind": "fitness", "negate": True},
    ),
    "forest-fused": (
        {"surface": "batch_forward", "kind": "value"},
        {"surface": "sr_fitness", "kind": "fitness", "negate": True},
    ),
}

#: The speedup baselines of the report: ``(label, lane, surface)``.
BASELINES: Tuple[Tuple[str, str, str], ...] = (
    ("numpy", "numpy", "evaluate"),
    ("torch-batched", "torch-batched", "batch_forward"),
    ("torch-per-tree", "torch-per-tree", "forward"),
    ("forest bf", "forest-auto", "batch_forward"),
    ("forest SR", "forest-auto", "sr_fitness"),
)


# --------------------------------------------------------------------------- #
# lazy imports (the ONE torch/evoxc boundary, resolved at CALL time)
# --------------------------------------------------------------------------- #
def _tree_module():
    """The torch stack's top-level module, imported on first use.

    Importing it is what fixes the evaluation ROUTE for the rest of the process
    (``dense_engine.resolve_route`` memoizes), which is why the ``forest-fused``
    control has to run in its own child process with the variable already set.
    """
    return importlib.import_module("evogp.tree")


def _cuda() -> str:
    """The torch device string the torch stack hard-codes."""
    return "cuda"


def scope_reason(lane: str, pop: int) -> Optional[str]:
    """Why ``(lane, pop)`` is out of this round's scope, or ``None`` if it is in it."""
    ceiling = SIDE_MAX_POP.get(lane)
    if ceiling is None or int(pop) <= int(ceiling):
        return None
    return (
        f"{lane} is limited to pop <= {ceiling} this round "
        f"(a python loop over {int(pop)} trees is ~0.7 s per call at pop 8192)"
    )


# --------------------------------------------------------------------------- #
# the shared population: ONE set of trees, every layout, setup never timed
# --------------------------------------------------------------------------- #
class _Population:
    """ONE ``evogp.gp`` population in every layout the lanes need.

    Built on the numpy backend so all lanes can be fed from the SAME rows: the
    ``GPState`` (the numpy lane and its ``predict`` oracle), the legacy triple
    (:func:`evogp.gp.encoding.to_legacy_triple` -> the torch ``Forest`` and the
    plain-torch evaluators) and, through :meth:`upload`, the device channels the
    two ``evogp.gp`` device lanes read.
    """

    def __init__(self, pop: int, rows: int = ROWS, seed: int = SEED) -> None:
        self.tree, self.gp_config, self.x, self.y = make_configs(pop, rows, seed)
        self.pop, self.rows, self.seed = int(pop), int(rows), int(seed)
        self.input_len = int(self.x.shape[1])
        self.state = init_population(
            (self.tree, self.gp_config), seed, backend="numpy"
        )
        self.legacy = encoding.to_legacy_triple(self.state)
        # the ONE channel parser of the stack, so the uploaded rows are exactly
        # the rows the numpy lane evaluates
        self.arrays = [
            np.array(array, copy=True)
            for array in encoding._channel_arrays(self.state)  # noqa: SLF001
        ]
        self.device_channels: Optional[Tuple[Any, ...]] = None
        self.upload_check: Optional[Dict[str, Any]] = None

    def upload(self, device) -> Tuple[Any, ...]:
        """Upload the channels ONCE and verify the round trip bit-identically."""
        evoxc = _evoxc()
        self.device_channels = tuple(
            evoxc.from_numpy(np.ascontiguousarray(array)).to(device)
            for array in self.arrays
        )
        cpu = evoxc.core.Device("cpu", 0)
        back = [np.array(tensor.to(cpu).numpy(), copy=True) for tensor in self.device_channels]
        identical = all(
            np.array_equal(left, right)
            for left, right in zip(self.arrays, back)
        )
        self.upload_check = {
            "identical": bool(identical),
            "channels": len(self.arrays),
            "pop": self.pop,
            "max_nodes": int(self.arrays[0].shape[1]),
            "note": "device channels read back and compared to the numpy rows",
        }
        if not identical:
            raise RuntimeError(
                "the uploaded device channels are not bit-identical to the numpy "
                "population; every device lane would evaluate different trees"
            )
        return self.device_channels

    def torch_dataset(self, torch):
        """``(x, labels)`` on the torch device, the data every torch lane sees."""
        device = _cuda()
        x = torch.as_tensor(
            np.ascontiguousarray(self.x), device=device, dtype=torch.float32
        )
        labels = torch.as_tensor(
            np.ascontiguousarray(self.y.reshape(-1, 1)), device=device,
            dtype=torch.float32,
        )
        return x, labels

    def forest(self, torch):
        """The torch stack's ``Forest`` over this population's legacy triple."""
        tree = _tree_module()
        values, node_type, subtree_size = self.legacy
        device = _cuda()
        return tree.Forest(
            self.input_len,
            1,
            torch.as_tensor(np.ascontiguousarray(values), device=device,
                            dtype=torch.float32),
            torch.as_tensor(np.ascontiguousarray(node_type), device=device,
                            dtype=torch.int16),
            torch.as_tensor(np.ascontiguousarray(subtree_size), device=device,
                            dtype=torch.int16),
        )


# --------------------------------------------------------------------------- #
# the plain-torch lanes (the benchmark campaign's two baseline idioms)
# --------------------------------------------------------------------------- #
def _safe_int(value: float) -> int:
    return 0 if not np.isfinite(value) else int(np.trunc(float(value)))


def _kind_from_type(node_type: int) -> int:
    """The engine's if/else chain: 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, else ternary."""
    if node_type in (0, 1, 2, 3):
        return int(node_type)
    return 4


def _arity_of(function_id: int) -> int:
    if function_id == 0:  # IF
        return 3
    if function_id <= 12:  # LT .. GE
        return 2
    return 1


#: Arity of a node KIND (``VAR``/``CONST`` 0, ``UFUNC`` 1, ``BFUNC`` 2, ternary 3).
_KIND_ARITY = {2: 1, 3: 2, 4: 3}

#: ``LOOSE_LOG(0)`` returns -MAX_VAL and ``LOOSE_DIV``/``LOOSE_INV`` clamp to DELTA.
_MAX_VAL = 1e9
_DELTA = 1e-9


def _kind_arity(kind: int) -> int:
    return _KIND_ARITY.get(int(kind), 0)


def _torch_ops(torch) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    """``(unary, binary, compare)`` op tables of the plain-torch baseline (by id)."""
    t = torch
    unary = {14: t.sin, 15: t.cos, 16: t.tan, 17: t.sinh, 18: t.cosh, 19: t.tanh,
             20: t.log, 22: t.exp, 25: t.neg, 26: t.abs, 27: t.sqrt}
    binary = {1: t.add, 2: t.sub, 3: t.mul, 6: t.pow}
    compare = {10: t.lt, 11: t.gt, 12: t.le, 13: t.ge}
    return unary, binary, compare


class _WholePopulationEvaluator:
    """The plain-torch batched baseline: the whole population in ONE pass.

    Copied idiom (not imported) from the benchmark campaign's ``BatchedEvaluator``,
    which is the best a pure-torch implementation does: every node position of the
    prefix-encoded forest becomes a ``(pop, rows)`` value in a position-indexed
    buffer, children are fetched per row by index (no python loop over trees), and
    the node kind / function id is resolved with per-``(kind, function)`` masks.
    Only the candidates that occur at a position are computed, so a call costs
    ``max_tree_len`` positions x (few ops) whatever the population.

    Semantics are the ones the engine documents (``../tree/ENGINE_SPEC.md``):
    ``DIV`` masks a zero divisor with NaN, ``MAX``/``MIN`` compare-select,
    comparisons yield +-1, ``IF`` selects on ``cond > 0``; the single-output
    forests of this campaign never reach the ternary path in practice but it is
    implemented, so a tree that does is measured, not mis-measured.

    The torch module is passed in (never imported at this module's scope) and the
    whole per-position plan is built in ``__init__``, i.e. setup is never timed.
    """

    def __init__(self, torch, forest) -> None:
        self._t = torch
        self.device = torch.device(_cuda())
        values, types, sizes = _host_rows(forest)
        self.pop_size, self.max_tree_len = values.shape
        self.output_len = int(forest.output_len)
        self.input_len = int(forest.input_len)
        if self.output_len != 1:
            raise NotImplementedError("the plain-torch lanes cover single-output forests")
        self._constants: Dict[float, Any] = {}
        self._nan = self._scalar(float("nan"))
        self._unary_ops, self._binary_ops, self._compare_ops = _torch_ops(torch)

        pop, length = self.pop_size, self.max_tree_len
        self.kind = np.vectorize(_kind_from_type, otypes=[np.int32])(types)
        self.active = np.arange(length)[None, :] < sizes[:, 0][:, None]
        sizes_p = np.concatenate([sizes, np.zeros((pop, 1), dtype=np.int64)], axis=1)
        rows = np.arange(pop)
        self.child1 = np.minimum(np.arange(1, length + 1), length)
        self.child2: List[Any] = []
        self.child3: List[Any] = []
        for index in range(length):
            first = int(self.child1[index])
            first_size = sizes_p[:, first] if first < length else np.zeros(pop, np.int64)
            second = np.minimum(first + first_size, length)
            second_size = sizes_p[rows, np.minimum(second, length)]
            third = np.minimum(second + second_size, length)
            self.child2.append(self._index(second))
            self.child3.append(self._index(third))
        self.plan: List[Dict[str, Any]] = []
        for index in range(length):
            kinds = self.kind[:, index]
            live = self.active[:, index]
            step: Dict[str, Any] = {"index": index, "cases": []}
            if np.any(live & (kinds == 0)):
                step["var_mask"] = self._mask(live & (kinds == 0))
                step["var_index"] = self._t.as_tensor(
                    np.array([_safe_int(value) for value in values[:, index]]),
                    device=self.device,
                ).clamp_(0, self.input_len - 1)
            if np.any(live & (kinds == 1)):
                step["const_mask"] = self._mask(live & (kinds == 1))
                step["const_value"] = self._t.as_tensor(
                    np.asarray(values[:, index], dtype=np.float32), device=self.device
                ).view(pop, 1)
            for kind in (2, 3, 4):
                base = live & (kinds == kind)
                if not np.any(base):
                    continue
                ids = np.array([_safe_int(value) for value in values[:, index]])
                for fid in sorted({int(value) for value in ids[base]}):
                    valid = base & (ids == fid) & (_arity_of(fid) == _kind_arity(kind))
                    if not np.any(valid):
                        continue
                    step["cases"].append({"kind": kind, "fid": fid,
                                          "mask": self._mask(valid)})
            self.plan.append(step)

    # -- setup helpers -------------------------------------------------------
    def _scalar(self, value: float):
        return self._t.tensor(float(value), dtype=self._t.float32, device=self.device)

    def _mask(self, boolean):
        return self._t.as_tensor(boolean, device=self.device).view(self.pop_size, 1)

    def _index(self, values):
        array = np.asarray(values, dtype=np.int64)
        if int(array.min()) == int(array.max()):
            return int(array[0])
        return self._t.as_tensor(array, device=self.device, dtype=self._t.int64)

    def _constant(self, value: float):
        key = float(value)
        tensor = self._constants.get(key)
        if tensor is None:
            tensor = self._constants[key] = self._scalar(key)
        return tensor

    def _child(self, buffer, index, rows):
        if isinstance(index, int):
            return buffer[:, index]
        return buffer[rows, index]

    # -- the ops (float32, plain torch) --------------------------------------
    def _unary(self, fid: int, a):
        t = self._t
        if fid == 21:  # LOOSE_LOG: log|a| with -MAX_VAL at zero
            return t.where(a == 0.0, self._scalar(-_MAX_VAL), t.log(t.abs(a)))
        if fid == 23:  # INV masks a zero input with NaN
            return t.where(a == 0.0, self._nan, t.reciprocal(a))
        if fid == 24:  # LOOSE_INV clamps |a| to DELTA (copysign)
            clamped = t.where(t.abs(a) <= _DELTA, t.copysign(self._scalar(_DELTA), a), a)
            return t.reciprocal(clamped)
        if fid == 28:  # LOOSE_SQRT takes |a| when a <= 0
            return t.sqrt(t.where(a <= 0.0, t.abs(a), a))
        operator = self._unary_ops.get(int(fid))
        return operator(a) if operator is not None else t.zeros_like(a)

    def _binary(self, fid: int, a, b):
        t = self._t
        if fid == 4:  # DIV masks a zero divisor with NaN
            return t.where(b == 0.0, self._nan, a / b)
        if fid == 5:  # LOOSE_DIV clamps |b| to DELTA (copysign)
            clamped = t.where(t.abs(b) <= _DELTA, t.copysign(self._scalar(_DELTA), b), b)
            return a / clamped
        if fid == 7:  # LOOSE_POW of (0, 0) is 0 and the base is |a|
            return t.where((a == 0.0) & (b == 0.0), self._scalar(0.0), t.pow(t.abs(a), b))
        if fid == 8:  # MAX selects (NaN goes right, like the engine)
            return t.where(a >= b, a, b)
        if fid == 9:  # MIN selects
            return t.where(a <= b, a, b)
        compare = self._compare_ops.get(int(fid))
        if compare is not None:  # LT/GT/LE/GE return +-1, never a boolean
            return t.where(compare(a, b), self._scalar(1.0), self._scalar(-1.0))
        operator = self._binary_ops.get(int(fid))
        return operator(a, b) if operator is not None else t.zeros_like(a)

    # -- the one pass --------------------------------------------------------
    def _evaluate(self, get_var: Callable[[int], Any], n: int):
        pop, length = self.pop_size, self.max_tree_len
        rows = self._t.arange(pop, device=self.device)
        buffer = self._t.zeros(
            (pop, length + 1, n), dtype=self._t.float32, device=self.device
        )
        for step in reversed(self.plan):
            index = step["index"]
            value = self._t.zeros((pop, n), dtype=self._t.float32, device=self.device)
            if "const_mask" in step:
                value = self._t.where(step["const_mask"], step["const_value"], value)
            if "var_mask" in step:
                value = self._t.where(step["var_mask"], get_var(index), value)
            first = int(self.child1[index])
            for case in step["cases"]:
                kind, fid = case["kind"], case["fid"]
                if kind == 2:
                    candidate = self._unary(fid, buffer[:, first])
                elif kind == 3:
                    candidate = self._binary(
                        fid, buffer[:, first],
                        self._child(buffer, self.child2[index], rows),
                    )
                else:
                    candidate = self._t.where(
                        buffer[:, first] > 0.0,
                        self._child(buffer, self.child2[index], rows),
                        self._child(buffer, self.child3[index], rows),
                    )
                value = self._t.where(case["mask"], candidate, value)
            buffer[:, index] = value
        return buffer[:, 0]

    def batch_forward(self, x):
        """``(rows, input_len)`` -> ``(pop, rows)`` (single-output forests)."""

        def get_var(index: int):
            return self._t.index_select(x, 1, self.plan[index]["var_index"]).transpose(0, 1)

        with self._t.no_grad():
            return self._evaluate(get_var, int(x.shape[0]))


def _host_rows(forest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(value float32, type int32, size int32)`` of a ``Forest``, on the host."""
    torch = _torch()
    values = forest.batch_node_value.detach().to("cpu", torch.float32).numpy()
    types = forest.batch_node_type.detach().to("cpu", torch.int16).numpy().astype(np.int32)
    sizes = forest.batch_subtree_size.detach().to("cpu", torch.int16).numpy().astype(np.int32)
    return (
        np.ascontiguousarray(values),
        np.ascontiguousarray(types),
        np.ascontiguousarray(sizes),
    )


def _per_tree_forward(forest, x):
    """The plain-torch per-tree walk: one ``Tree.forward`` per tree, stacked.

    ``forest[i]`` is a ``Tree`` whose own evaluation is the engine's population-of-
    one path -- the honest "evaluate each tree with torch" route, one cluster of
    ops per node per tree.
    """
    rows = int(x.shape[0])
    values = [forest[index].forward(x).reshape(rows) for index in range(forest.pop_size)]
    return _torch().stack(values, dim=0)


# --------------------------------------------------------------------------- #
# one lane, one population
# --------------------------------------------------------------------------- #
def _surfaces(lane: str) -> Tuple[Dict[str, Any], ...]:
    return LANE_SURFACES[lane]


def _surface_record(spec: Dict[str, Any], cold: Dict[str, Any], timing: Dict[str, Any]):
    return {
        "surface": spec["surface"],
        "kind": spec["kind"],
        "negate": bool(spec.get("negate")),
        "cold_ms": cold["median_ms"],
        "timing": timing,
    }


def _lane_numpy(population: _Population, *, warmup: int, reps: int):
    """:class:`evogp.gp.problem.SymbolicRegression` on the host (the reference)."""
    host = problem.SymbolicRegression(population.x, population.y)
    state = population.state
    values: Dict[str, np.ndarray] = {}
    surfaces: Dict[str, Dict[str, Any]] = {}
    cold, _ = _median(lambda: host.evaluate(state), 1, 0)
    timing, fitness = _median(lambda: host.evaluate(state), reps, warmup)
    surfaces["evaluate"] = _surface_record(_surfaces("numpy")[0], cold, timing)
    values["evaluate"] = np.array(fitness, dtype=np.float32, copy=True)

    cold, _ = _median(lambda: host.predict(state), 1, 0)
    timing, predictions = _median(lambda: host.predict(state), reps, warmup)
    surfaces["predict"] = _surface_record(_surfaces("numpy")[1], cold, timing)
    values["predict"] = np.array(predictions, dtype=np.float32, copy=True)
    return (
        {
            "lane": "numpy",
            "surfaces": surfaces,
            "facts": {
                "backend": "numpy",
                "device": "cpu",
                "reference": "this lane is the fitness AND the value reference",
            },
        },
        values,
    )


def _lane_gp(population: _Population, *, kind: str, warmup: int, reps: int):
    """The device-resident in-graph route: device channels in, device fitness out."""
    evaluator = GraphEvaluator(
        population.x, population.y, used_ids=population.tree, backend="iree",
        device=kind,
    )
    channels = population.device_channels
    if channels is None:
        channels = population.upload(_evoxc().core.Device(kind, 0))
    cold, _ = _median(lambda: evaluator.evaluate(channels), 1, 0, _sync)
    timing, fitness = _median(lambda: evaluator.evaluate(channels), reps, warmup, _sync)
    values = {"evaluate": readback_fitness(fitness, population.pop)}
    return (
        {
            "lane": "gp",
            "surfaces": {"evaluate": _surface_record(_surfaces("gp")[0], cold, timing)},
            "facts": {
                "backend": "iree",
                "device": kind,
                "used_ids": "the run's GPTreeConfig (the narrow dispatch chain)",
                "readback": "one explicit D2H AFTER the timed block, for parity only",
            },
        },
        values,
    )


def _lane_gp_fused(population: _Population, *, kind: str, warmup: int, reps: int):
    """The device-native fused route (zero-copy DLPack views into the Triton kernel)."""
    from . import fused_eval  # noqa: PLC0415 - local: it owns the torch boundary

    evaluator = fused_eval.FusedDeviceEvaluator(population.x, population.y)
    channels = population.device_channels
    if channels is None:
        channels = population.upload(_evoxc().core.Device(kind, 0))
    cold, _ = _median(lambda: evaluator.evaluate(channels), 1, 0, _sync)
    timing, fitness = _median(lambda: evaluator.evaluate(channels), reps, warmup, _sync)
    values = {"evaluate": readback_fitness(fitness, population.pop)}
    return (
        {
            "lane": "gp-fused",
            "surfaces": {"evaluate": _surface_record(_surfaces("gp-fused")[0], cold, timing)},
            "facts": {
                "route": evaluator.last_route,
                "route_counts": {key: int(value) for key, value in evaluator.route_counts.items()},
                "last_fallback": evaluator.last_fallback,
                "available": bool(evaluator.available()),
                "device_available": bool(evaluator.device_available()),
                "interop_available": bool(fused_eval.interop_available(kind)),
                "triton_cache_dir": fused_eval.ensure_triton_cache_dir(),
                "note": (
                    "route=='device' means the call was served by the kernel on "
                    "exported evoxc device buffers, not by the host hand-off"
                ),
            },
        },
        values,
    )


def _lane_torch_batched(population: _Population, *, warmup: int, reps: int):
    """The plain-torch whole-population pass (the batched baseline)."""
    torch = _torch()
    forest = population.forest(torch)
    x, _labels = population.torch_dataset(torch)
    evaluator = _WholePopulationEvaluator(torch, forest)
    cold, _ = _median(lambda: evaluator.batch_forward(x), 1, 0, _sync)
    timing, predictions = _median(lambda: evaluator.batch_forward(x), reps, warmup, _sync)
    values = {"batch_forward": predictions.detach().to("cpu", torch.float32).numpy()
              .reshape(population.pop, -1)}
    return (
        {
            "lane": "torch-batched",
            "surfaces": {
                "batch_forward": _surface_record(
                    _surfaces("torch-batched")[0], cold, timing
                )
            },
            "facts": {
                "kind": "value",
                "route": "plain torch, one vectorized pass, no evoxc",
                "setup": "the per-position plan is built in the constructor (never timed)",
            },
        },
        values,
    )


def _lane_torch_per_tree(population: _Population, *, warmup: int, reps: int):
    """The plain-torch per-tree walk (one ``Tree.forward`` per tree, stacked)."""
    torch = _torch()
    forest = population.forest(torch)
    x, _labels = population.torch_dataset(torch)
    cold, _ = _median(lambda: _per_tree_forward(forest, x), 1, 0, _sync)
    timing, predictions = _median(
        lambda: _per_tree_forward(forest, x), reps, warmup, _sync
    )
    values = {"forward": predictions.detach().to("cpu", torch.float32).numpy()}
    return (
        {
            "lane": "torch-per-tree",
            "surfaces": {"forward": _surface_record(_surfaces("torch-per-tree")[0], cold, timing)},
            "facts": {
                "kind": "value",
                "route": "plain torch, python loop over trees, Tree.forward each",
                "scope": SIDE_MAX_POP["torch-per-tree"],
            },
        },
        values,
    )


def _lane_forest(population: _Population, *, lane: str, warmup: int, reps: int):
    """The torch stack's own ``Forest`` under the shipped default or forced fused route.

    ``EVOGP_EVAL_MODE`` is set (or dropped) BEFORE the torch stack is imported:
    ``dense_engine.resolve_route`` memoizes its decision, so flipping the variable
    later has no effect and the ``forest-fused`` control therefore runs in its own
    child process.
    """
    from . import fused_eval  # noqa: PLC0415 - local: the triton cache directory

    eval_mode = "fused" if lane == "forest-fused" else "auto"
    if eval_mode == "auto":
        os.environ.pop("EVOGP_EVAL_MODE", None)
    else:
        os.environ["EVOGP_EVAL_MODE"] = eval_mode
    if eval_mode != "auto":
        fused_eval.ensure_triton_cache_dir()

    torch = _torch()
    forest = population.forest(torch)
    x, labels = population.torch_dataset(torch)
    route = importlib.import_module("evogp.tree.dense_engine").resolve_route(
        forest.pop_size
    )
    specs = {spec["surface"]: spec for spec in _surfaces(lane)}
    values: Dict[str, np.ndarray] = {}
    surfaces: Dict[str, Dict[str, Any]] = {}

    cold, _ = _median(lambda: forest.batch_forward(x), 1, 0, _sync)
    timing, predictions = _median(lambda: forest.batch_forward(x), reps, warmup, _sync)
    surfaces["batch_forward"] = _surface_record(specs["batch_forward"], cold, timing)
    values["batch_forward"] = (
        predictions.detach().to("cpu", torch.float32).numpy()
        .reshape(population.pop, -1)
    )

    cold, _ = _median(lambda: forest.SR_fitness(x, labels), 1, 0, _sync)
    timing, loss = _median(lambda: forest.SR_fitness(x, labels), reps, warmup, _sync)
    surfaces["sr_fitness"] = _surface_record(specs["sr_fitness"], cold, timing)
    # the RAW surface output: ``SR_fitness`` returns the POSITIVE loss, and the
    # ``negate`` flag of the surface spec is what turns it into the fitness the
    # pipeline consumes -- one place applies the sign, so the parity step and the
    # record can never disagree about it
    values["sr_fitness"] = np.array(
        loss.detach().to("cpu", torch.float32).numpy(), dtype=np.float32, copy=True
    ).reshape(population.pop)
    return (
        {
            "lane": lane,
            "surfaces": surfaces,
            "facts": {
                "eval_mode": eval_mode,
                "route": route.engine,
                "backend": route.backend,
                "fallback_backend": route.fallback_backend,
                "note": (
                    "route is what dense_engine.resolve_route answered in this "
                    "process; 'fused' means the shipped default served the call"
                ),
            },
        },
        values,
    )


_LANE_RUNNERS: Dict[str, Callable[..., Tuple[Dict[str, Any], Dict[str, np.ndarray]]]] = {
    "numpy": _lane_numpy,
    "gp": _lane_gp,
    "gp-fused": _lane_gp_fused,
    "torch-batched": _lane_torch_batched,
    "torch-per-tree": _lane_torch_per_tree,
}


# --------------------------------------------------------------------------- #
# one population, every requested lane (child mode)
# --------------------------------------------------------------------------- #
def measure_pop(
    pop: int,
    rows: int = ROWS,
    *,
    lanes: Sequence[str] = LANES,
    warmup: int = WARMUP,
    reps: int = REPS,
    seed: int = SEED,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure every requested lane at ONE population size, in this process.

    One :class:`_Population` feeds every lane, so all of them evaluate the same
    trees.  ``forest-fused`` is the one lane that cannot share a process with
    another (it forces ``EVOGP_EVAL_MODE=fused`` before the torch stack is
    imported), so it is refused here when mixed and scheduled as its own child.
    """
    lanes = list(lanes)
    unknown = [lane for lane in lanes if lane not in LANES]
    if unknown:
        raise ValueError(f"unknown lanes {unknown}; known: {', '.join(LANES)}")
    if "forest-fused" in lanes and len(lanes) > 1:
        raise ValueError(
            "forest-fused forces EVOGP_EVAL_MODE=fused before evogp.tree is "
            "imported, so it must be measured alone in its own process"
        )
    kind = kind or _device_kind()
    population = _Population(int(pop), int(rows), int(seed))
    if any(lane in ("gp", "gp-fused") for lane in lanes):
        population.upload(_evoxc().core.Device(kind, 0))

    # the reference of BOTH parity groups: one evaluated generation of the host
    # lane, computed even when the numpy lane itself is not timed
    reference = problem.SymbolicRegression(population.x, population.y)
    reference_fitness = np.array(
        reference.evaluate(population.state), dtype=np.float32, copy=True
    )
    reference_values = np.array(
        reference.predict(population.state), dtype=np.float32, copy=True
    )

    records: List[Dict[str, Any]] = []
    scoped: List[Dict[str, Any]] = []
    parity: Dict[str, Dict[str, Any]] = {"fitness": {}, "value": {}}
    for lane in lanes:
        reason = scope_reason(lane, population.pop)
        if reason is not None:
            print(f"# skipping {lane} (scoped out: {reason})", flush=True)
            scoped.append({"lane": lane, "pop": population.pop, "rows": population.rows,
                           "scoped_out": True, "reason": reason})
            continue
        print(f"# {lane} pop={population.pop} rows={population.rows}", flush=True)
        if lane in ("forest-auto", "forest-fused"):
            record, values = _lane_forest(
                population, lane=lane, warmup=int(warmup), reps=int(reps)
            )
        else:
            runner = _LANE_RUNNERS[lane]
            kwargs: Dict[str, Any] = {"warmup": int(warmup), "reps": int(reps)}
            if lane in ("gp", "gp-fused"):
                kwargs["kind"] = kind
            record, values = runner(population, **kwargs)
        records.append(record)
        for spec in _surfaces(lane):
            name = spec["surface"]
            measured = values.get(name)
            if measured is None:
                continue
            signed = -measured if spec.get("negate") else measured
            target = "fitness" if spec["kind"] == "fitness" else "value"
            baseline = reference_fitness if target == "fitness" else reference_values
            entry = _parity(np.asarray(baseline), np.asarray(signed))
            entry["lane"] = lane
            entry["surface"] = name
            entry["kind"] = spec["kind"]
            if spec.get("negate"):
                entry["note"] = "the measured loss was negated before the comparison"
            parity[target][f"{lane}.{name}"] = entry

    return {
        "pop": population.pop,
        "rows": population.rows,
        "seed": population.seed,
        "device": kind,
        "upload": population.upload_check,
        "records": records,
        "parity": parity,
        "scoped_out": scoped,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _warm_ms(payload: Dict[str, Any], lane: str, surface: str) -> Optional[float]:
    """The warm median (ms) of one lane surface in one population payload."""
    for record in payload.get("records") or []:
        if record.get("lane") != lane:
            continue
        block = (record.get("surfaces") or {}).get(surface)
        if block and block.get("timing"):
            return float(block["timing"]["median_ms"])
    return None


def _ratio(base: Optional[float], value: Optional[float]) -> str:
    if base is None or value is None or not value:
        return "-"
    return f"{base / value:.2f}x"


def _number(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3g}"


def report(
    payloads: Sequence[Dict[str, Any]],
    scoped_out: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Every lane's per-population per-call timing and the two parity groups."""
    scoped = [entry for entry in (scoped_out or [])]
    for payload in payloads:
        scoped.extend(payload.get("scoped_out") or [])
    payloads = [payload for payload in payloads if payload.get("records")]
    present = [lane for lane in LANES
               if any(record.get("lane") == lane
                      for payload in payloads for record in payload.get("records") or [])]
    lines = ["## per-call timing (ms per evaluation call, warm medians)", ""]
    for lane in present:
        header = " | ".join(
            ["pop", "rows", "surface", "cold", "warm"]
            + [label for label, _lane, _surface in BASELINES]
        )
        lines.append(f"### `{lane}`")
        lines.append("")
        lines.append(f"| {header} |")
        lines.append(
            "|---:|---:|---|---:|---:|"
            + "".join("---:|" for _ in BASELINES)
        )
        for payload in sorted(payloads, key=lambda item: item["pop"]):
            for record in payload.get("records") or []:
                if record.get("lane") != lane:
                    continue
                for surface, block in (record.get("surfaces") or {}).items():
                    warm = block["timing"]["median_ms"] if block.get("timing") else None
                    cells = [
                        str(payload["pop"]),
                        str(payload["rows"]),
                        f"{surface} ({block.get('kind')})",
                        "-" if block.get("cold_ms") is None else f"{block['cold_ms']:.3f}",
                        "-" if warm is None else f"{warm:.3f}",
                    ]
                    for _label, base_lane, base_surface in BASELINES:
                        cells.append(
                            _ratio(
                                _warm_ms(payload, base_lane, base_surface),
                                warm,
                            )
                        )
                    lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    parity_lines: List[str] = []
    for payload in sorted(payloads, key=lambda item: item["pop"]):
        for group, title in (
            ("fitness", "fitness vs `numpy.evaluate`"),
            ("value", "value vs `numpy.predict`"),
        ):
            entries = (payload.get("parity") or {}).get(group) or {}
            if not entries:
                continue
            parity_lines.append(
                f"**{group}** ({title}) -- pop {payload['pop']} rows {payload['rows']}"
            )
            parity_lines.append("")
            for name in sorted(entries):
                entry = entries[name]
                parity_lines.append(
                    "- {name}: max_rel {rel}, scaled {scaled}, "
                    "{mismatch} finiteness mismatches ({compared} comparable)".format(
                        name=name,
                        rel=_number(entry["max_rel"]),
                        scaled=_number(entry["max_rel_scaled"]),
                        mismatch=entry["finiteness_mismatch"],
                        compared=entry["comparable"],
                    )
                )
            parity_lines.append("")
    if parity_lines:
        lines.append(
            "## parity (both groups against the `numpy` lane, same trees)"
        )
        lines.append("")
        lines.extend(parity_lines)

    if scoped:
        lines.append("## scoped out")
        lines.append("")
        for entry in scoped:
            lines.append(
                "- {lane} pop {pop} rows {rows}: {reason}".format(
                    lane=entry.get("lane") or entry.get("side"),
                    pop=entry.get("pop"),
                    rows=entry.get("rows"),
                    reason=entry.get("reason", "out of this round's scope"),
                )
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# orchestration: one process per population
# --------------------------------------------------------------------------- #
def _child_command(pop: int, rows: int, lanes: Sequence[str], args, fragment: str):
    return [
        sys.executable, "-m", "evogp.gp.measure_popscale",
        "--pop", str(pop), "--rows", str(rows), "--lanes", ",".join(lanes),
        "--warmup", str(args.warmup), "--reps", str(args.reps),
        "--seed", str(args.seed), "--json", fragment,
    ]


def _command_line(raw: Sequence[str]) -> str:
    return " ".join([sys.executable, "-m", "evogp.gp.measure_popscale"] + list(raw))


def _child(args) -> int:
    """Child mode: measure the requested lanes of ONE population in this process."""
    lanes = [
        text.strip() for text in str(args.lanes).split(",") if text.strip()
    ]
    if args.lane:
        lanes = [args.lane]
    elif len(lanes) > 1 and "forest-fused" in lanes:
        # the control is scheduled as its OWN child (it forces the route before
        # the torch stack is imported), so it never shares this process
        lanes = [lane for lane in lanes if lane != "forest-fused"]
    state = None if args.no_guard else guard(args.guard_wait)
    payload = measure_pop(
        int(args.pop), int(args.rows), lanes=lanes, warmup=int(args.warmup),
        reps=int(args.reps), seed=int(args.seed),
    )
    record = {
        "command": _command_line(args.raw),
        "config": {
            "workload": dict(WORKLOAD),
            "gp_knobs": dict(GP_KNOBS),
            "pop": int(args.pop),
            "rows": int(args.rows),
            "lanes": lanes,
            "warmup": int(args.warmup),
            "reps": int(args.reps),
            "seed": int(args.seed),
            "side_max_pop": dict(SIDE_MAX_POP),
        },
        "guard": state,
        "versions": versions(),
        "pop": payload,
    }
    if args.json:
        _write(args.json, record)
    print(json.dumps(record, indent=2, sort_keys=True, default=float))
    return 0


def _parent(args) -> int:
    """Parent mode: one child per population (sequential), print and merge."""
    pops = [int(text) for text in str(args.pops).split(",") if text.strip()]
    lanes = [text.strip() for text in str(args.lanes).split(",") if text.strip()]
    unknown = [lane for lane in lanes if lane not in LANES]
    if unknown:
        raise SystemExit(f"unknown --lanes entries {unknown}; known: {', '.join(LANES)}")
    control_pop = int(args.fused_control_pop) if args.fused_control_pop else max(pops)
    fragment_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"evogp-measure-popscale-{os.getpid()}"
    )
    os.makedirs(fragment_dir, exist_ok=True)
    payloads: List[Dict[str, Any]] = []
    guards: List[Dict[str, Any]] = []
    children: List[Dict[str, Any]] = []
    schedule = [
        (pop, [lane for lane in lanes if lane != "forest-fused"])
        for pop in pops
    ]
    if "forest-fused" in lanes:
        if args.skip_fused_control:
            print("# fused control skipped (--skip-fused-control)", flush=True)
        elif control_pop not in pops:
            print(
                f"# fused control not scheduled (--fused-control-pop {control_pop} "
                f"is not in --pops)",
                flush=True,
            )
        else:
            schedule.append((control_pop, ["forest-fused"]))
    for pop, lane_list in schedule:
        label = f"pop={pop} rows={args.rows} lanes={','.join(lane_list)}"
        if not lane_list:
            continue
        state = None if args.no_guard else guard(args.guard_wait)
        if state is not None:
            guards.append({"pop": pop, "rows": int(args.rows), **state})
            if not state.get("ok"):
                print(f"# card busy before {label} "
                      f"(gpu use {state.get('gpu_use_pct')}); measuring anyway",
                      flush=True)
        fragment = os.path.join(fragment_dir, f"{pop}-{'_'.join(lane_list)}.json")
        command = _child_command(pop, int(args.rows), lane_list, args, fragment)
        print(f"# measuring {label}", flush=True)
        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.time() - started
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout + completed.stderr)
            raise SystemExit(f"{label} failed (exit {completed.returncode})")
        with open(fragment, encoding="utf-8") as handle:
            payload = json.load(handle)
        payloads.append(payload["pop"])
        children.append({"pop": pop, "rows": int(args.rows), "lanes": lane_list,
                         "command": command, "wall_s": round(elapsed, 2)})
        print(f"#   done in {elapsed:.1f} s", flush=True)
    print()
    print(report(payloads))
    if args.json:
        scoped = [entry for payload in payloads for entry in payload.get("scoped_out") or []]
        from . import fused_eval  # noqa: PLC0415 - a capability probe, never a call

        try:
            interop = bool(fused_eval.interop_available(_device_kind()))
        except Exception as exc:  # noqa: BLE001 - provenance only
            interop = f"probe failed: {exc!r}"
        _write(
            args.json,
            {
                "command": _command_line(args.raw),
                "config": {
                    "workload": dict(WORKLOAD), "gp_knobs": dict(GP_KNOBS),
                    "pops": pops, "lanes": lanes, "rows": int(args.rows),
                    "warmup": int(args.warmup), "reps": int(args.reps),
                    "seed": int(args.seed),
                    "fused_control_pop": control_pop,
                    "skip_fused_control": bool(args.skip_fused_control),
                    "side_max_pop": dict(SIDE_MAX_POP),
                    "lane_surfaces": {lane: list(LANE_SURFACES[lane]) for lane in LANES},
                    "protocol": (
                        "one process per population; ONE gp population per pop "
                        "feeds every lane; cold = the first call (1 sample), warm = "
                        "the median of --reps timed calls after --warmup discarded "
                        "ones, torch.cuda.synchronize() around every timed "
                        "repetition; idle-device guard per child; forest-fused runs "
                        "in its own child with EVOGP_EVAL_MODE=fused set before "
                        "evogp.tree is imported"
                    ),
                },
                "versions": versions(),
                "git_head": _git_head(os.getcwd()),
                "interop_available": interop,
                "children": children,
                "guards": guards,
                "guard": {
                    "children": len(guards),
                    "all_ok": all(bool(state.get("ok")) for state in guards),
                    "skipped": bool(args.no_guard),
                },
                "scoped_out": scoped,
                "pops": payloads,
                "records": [record for payload in payloads
                            for record in payload.get("records") or []],
            },
        )
        print()
        print(f"# merged record written to {args.json}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: ``python -m evogp.gp.measure_popscale``."""
    parser = argparse.ArgumentParser(
        prog="python -m evogp.gp.measure_popscale",
        description=(
            "Per-call evaluation of ONE shared gp population across the "
            "population scale: the device-resident evogp.gp routes, the numpy "
            "reference, the plain-torch baselines and the torch stack."
        ),
    )
    for flag, options in (
        ("--pops", {"default": ",".join(str(pop) for pop in POPS),
                    "help": f"comma list of populations ({','.join(str(p) for p in POPS)})"}),
        ("--pop", {"type": int, "default": None, "help": "population (child mode)"}),
        ("--lanes", {"default": ",".join(LANES),
                     "help": f"comma list of lanes ({','.join(LANES)})"}),
        ("--lane", {"default": None,
                    "help": "measure only this lane (child mode; used for the control)"}),
        ("--rows", {"type": int, "default": ROWS, "help": f"data rows ({ROWS})"}),
        ("--reps", {"type": int, "default": REPS, "help": "timed repetitions"}),
        ("--warmup", {"type": int, "default": WARMUP,
                      "help": "discarded calls before the timed block"}),
        ("--seed", {"type": int, "default": SEED, "help": "RNG seed"}),
        ("--fused-control-pop", {"type": int, "default": None,
                                 "help": "population of the forest-fused control "
                                         "(default: the largest --pops entry)"}),
        ("--skip-fused-control", {"action": "store_true",
                                  "help": "do not schedule the forest-fused control"}),
        ("--no-guard", {"action": "store_true", "help": "skip the idle-device guard"}),
        ("--guard-wait", {"type": float, "default": 60.0,
                          "help": "seconds to wait for an idle card"}),
        ("--json", {"default": None, "help": "write the merged record to this path"}),
    ):
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    args.raw = list(argv if argv is not None else sys.argv[1:])
    if args.pop is not None:
        return _child(args)
    return _parent(args)


if __name__ == "__main__":  # pragma: no cover - a harness, never imported for effect
    raise SystemExit(main())
