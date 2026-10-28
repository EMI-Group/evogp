"""Check battery for the benchmark-support runner hooks (dev CLI, NOT pytest).

Runnable script (no ``test_`` prefix, no pytest import): it is the evidence
battery for the two frozen runner APIs documented in ``RUNNER_HOOKS.md`` --

    dense_engine.build_dense_runner(population_or_forest, inputs, *, ...)
    evoxc_engine.build_specialized_runner(population_or_forest, inputs, *, ...)

-- each returning an object whose repeatable ``.run()`` executes ONLY the
compiled forward pass (device-resident inputs -> device-resident result, no
readback, no host assembly, no re-plan / re-key / re-decode).

What it pins
------------
``Traffic`` monkeypatches every site a runner call can cross the host boundary
through and records, per call, the **caller's function name AND file** plus the
payload shape.  It mirrors ``device_resident_check.Traffic`` (the same sites,
the same "caller frame" idea); the file is added because the runner's entry
point is named ``run`` -- a name evoxc's own internals could reuse -- so a
name-only filter cannot separate the runner's calls from the backend's.

============================= ===================== ==========================
patched site                  direction             what it sees
============================= ===================== ==========================
``evoxc.core.Tensor.numpy``   evoxc -> host         every readback (filtered by caller frame; the numpy interpreter calls it internally thousands of times)
``evoxc.core.Tensor.to``      host -> evoxc device  every upload (device backends; a ``kind == "cpu"`` target is the READBACK and is not counted)
``dense_engine._pad_rows``    host only             every dense payload PREPARATION (all backends)
``torch.from_numpy``          host -> torch tensor  every host-side tensor assembly
``evoxc.run``                 graph invocation      every compiled-graph call
``evoxc_engine.build_cached`` compile               every compile-cache lookup
``evoxc_engine._executable``  memo lookup           every per-tree executable resolution
``evoxc_engine.decode_population`` host decode       every population decode
============================= ===================== ==========================

``evoxc.from_numpy`` is a ZERO-COPY HOST WRAP (it never transfers), so the
upload SITE is ``evoxc.core.Tensor.to`` -- only calls whose target kind is not
``"cpu"`` are uploads.  The caller-frame filter still separates the runner's and
the seam's calls (``payload`` / ``__init__`` / ``run`` ...) from the engine's and
the evoxc interpreter's own internals.

Stages
------
``dense``       ``build_dense_runner`` on a random population + one constant
                (never-compiled) tree: the build resolves ONE dense compile (one
                cache entry), stages the 7 node columns and the ``x`` block
                exactly once and performs one warm-up run with zero readbacks;
                every ``.run()`` then issues exactly ONE ``evoxc.run`` with zero
                readbacks / uploads / payload preparations / torch conversions /
                ``build_cached`` / ``_executable`` calls, repeats bit-exactly,
                and the cache counters do not move.  Numeric agreement vs
                ``evoxc_engine.evaluate`` and ``Forest.batch_forward``
                (``per_row=False``) and vs ``evaluate`` + ``Forest.forward``
                (``per_row=True``).  Warm per-call medians are printed.
``specialized`` ``build_specialized_runner`` on the same population: the build
                decodes once and resolves EVERY per-tree executable (one
                ``_executable`` lookup per non-constant tree, one ``build_cached``
                per DISTINCT program, zero runs), stages the input block once
                (``per_row=False``) or one row per tree (``per_row=True``); every
                ``.run()`` then issues exactly ``n_nonconst`` ``evoxc.run`` calls
                with zero readbacks / uploads / lookups / decodes, returns a
                ``None`` slot for each constant tree (``.constant_rows``), and
                repeats bit-exactly.  Numeric agreement vs ``evaluate`` on a
                single-output population (both ``per_row`` modes) and on a
                multi-output (``output_len=2`` OUT_NODE) population.
``edges``       the build-time error contracts (``DenseUnsupported`` for a
                multi-output population and for a chunked plan, ``ValueError``
                for a bad ``inputs`` shape / a ``per_row`` row mismatch, the
                specialized decoder's ``ValueError`` for a malformed tree,
                ``TypeError`` for a non-Forest-like container) and the fact that
                both runners IGNORE ``EVOGP_EVAL_MODE``: they are explicit
                route-selection APIs, not a new public route.

Usage (repo root, project venv)::

    PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      <venv>/bin/python src/evogp/tree/runner_check.py
    ... runner_check.py --stage dense,specialized --json $TMPDIR/runner.json
    ... runner_check.py --backend iree --gpu 0 --pop 4 --rows 8   # one config/process

Exit code 0 = every check passed, 1 = at least one failed, 2 = hard error.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import evoxc  # noqa: E402

from evogp.tree import Forest  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# Small helpers shared with the sibling batteries (same directory): the report
# object, the float32 comparison policy, the fixtures and the mode switch.
from evogp.tree.dense_integration_check import (  # noqa: E402
    Report,
    compare,
    multi_output_forest,
    pack,
    set_mode,
)
from evogp.tree.device_resident_check import (  # noqa: E402
    _backend,
    _nan_safe_equal,
    fixture,
)

_RTOL = 1e-4
_ATOL = 1e-4

_STAGES = ("dense", "specialized", "edges")

#: the dense VALUE variant stages exactly these many ``(L, P_chunk)`` node columns
_DENSE_COLUMNS = 7


# ---------------------------------------------------------------------------
# traffic counters
# ---------------------------------------------------------------------------

_D = "dense_engine.py"
_E = "evoxc_engine.py"

#: caller frames of the runners themselves (their ``__init__`` = the build phase).
_BUILD_FRAMES = (("__init__", _D), ("__init__", _E))
_RUN_FRAMES = (("run", _D), ("run", _E))
#: caller frames of the readback-free seam the dense runner runs through
#: (``payload`` is the dense payload-memo closure, i.e. the frame the
#: ``evoxc.core.Tensor.to`` upload is issued from once per distinct payload).
_SEAM_FRAMES = (
    ("payload", _D),
    ("_iter_plan_chunks", _D),
    ("_to_host", _D),
    ("run_plan", _D),
    ("run_loss_plan", _D),
)
#: caller frames of the READBACK-OWNING reference paths (the positive controls).
_REFERENCE_FRAMES = _SEAM_FRAMES + (
    ("_run_tree", _E),
    ("_evaluate_rows", _E),
    ("evaluate", _E),
    ("sr_fitness", _E),
)
_RUNNER_FRAMES = _BUILD_FRAMES + _RUN_FRAMES + _SEAM_FRAMES


def _frame(depth: int = 2) -> Tuple[str, str]:
    """``(function name, file name)`` of the frame that called a patched site."""
    frame = sys._getframe(depth)
    return frame.f_code.co_name, os.path.basename(frame.f_code.co_filename)


def _matching(records: Sequence[Any], frames: Sequence[Tuple[str, str]]) -> List[Any]:
    """Records whose first two elements (caller name, caller file) are in ``frames``."""
    wanted = set(frames)
    return [record for record in records if (record[0], record[1]) in wanted]


class Traffic:
    """Every host<->device crossing of one block, tagged with its call site.

    A context manager: it patches the conversion / compile / run helpers for the
    duration of the ``with`` block and restores them on exit (exceptions
    included).  Every record starts with ``(caller_name, caller_file)``; the
    payload shape follows where the site has one.
    """

    def __init__(self) -> None:
        self.readbacks: List[Tuple[str, str, Tuple[int, ...]]] = []
        self.uploads: List[Tuple[str, str, Tuple[int, ...]]] = []
        self.preparations: List[Tuple[str, str, Tuple[int, ...]]] = []
        self.torch_from_numpy: List[Tuple[str, str, Tuple[int, ...]]] = []
        self.runs: List[Tuple[str, str]] = []
        self.builds: List[Tuple[str, str]] = []
        self.lookups: List[Tuple[str, str]] = []
        self.decodes: List[Tuple[str, str]] = []

    def __enter__(self) -> "Traffic":
        original_numpy = evoxc.core.Tensor.numpy
        original_tensor_to = evoxc.core.Tensor.to
        original_pad_rows = D._pad_rows
        original_torch = torch.from_numpy
        original_run = evoxc.run
        original_build = E.build_cached
        original_lookup = E._executable
        original_decode = E.decode_population

        def tensor_numpy(self_: Any, *args: Any, **kwargs: Any):
            result = original_numpy(self_, *args, **kwargs)
            name, file = _frame()
            self.readbacks.append((name, file, tuple(self_.shape)))
            return result

        def tensor_to(self_: Any, device: Any, *args: Any, **kwargs: Any) -> Any:
            """The host -> device upload site (``.to(cpu)`` = readback, uncounted)."""
            result = original_tensor_to(self_, device, *args, **kwargs)
            if getattr(device, "kind", None) != "cpu":
                self.uploads.append((*_frame(), tuple(self_.shape)))
            return result

        def pad_rows(array: Any, rows: int) -> Any:
            self.preparations.append((*_frame(), tuple(np.shape(array))))
            return original_pad_rows(array, rows)

        def torch_from_numpy(array: Any, *args: Any, **kwargs: Any) -> Any:
            self.torch_from_numpy.append((*_frame(), tuple(np.shape(array))))
            return original_torch(array, *args, **kwargs)

        def run(executable: Any, *args: Any, **kwargs: Any) -> Any:
            self.runs.append(_frame())
            return original_run(executable, *args, **kwargs)

        def build_cached(*args: Any, **kwargs: Any) -> Any:
            self.builds.append(_frame())
            return original_build(*args, **kwargs)

        def executable(*args: Any, **kwargs: Any) -> Any:
            self.lookups.append(_frame())
            return original_lookup(*args, **kwargs)

        def decode_population(*args: Any, **kwargs: Any) -> Any:
            self.decodes.append(_frame())
            return original_decode(*args, **kwargs)

        evoxc.core.Tensor.numpy = tensor_numpy
        evoxc.core.Tensor.to = tensor_to
        D._pad_rows = pad_rows
        torch.from_numpy = torch_from_numpy
        evoxc.run = run
        E.build_cached = build_cached
        E._executable = executable
        E.decode_population = decode_population
        self._originals = (
            (evoxc.core.Tensor, "numpy", original_numpy),
            (evoxc.core.Tensor, "to", original_tensor_to),
            (D, "_pad_rows", original_pad_rows),
            (torch, "from_numpy", original_torch),
            (evoxc, "run", original_run),
            (E, "build_cached", original_build),
            (E, "_executable", original_lookup),
            (E, "decode_population", original_decode),
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        return None

    # -- views -------------------------------------------------------------
    def runner_readbacks(self) -> List[Tuple[str, str, Tuple[int, ...]]]:
        """Readbacks issued from a runner / seam frame (``_to_host`` etc.)."""
        return _matching(self.readbacks, _RUNNER_FRAMES)

    def reference_readbacks(self) -> List[Tuple[str, str, Tuple[int, ...]]]:
        """Readbacks issued by a readback-owning reference path (the control)."""
        return _matching(self.readbacks, _REFERENCE_FRAMES)

    def runner_torch_creations(self) -> List[Tuple[str, str, Tuple[int, ...]]]:
        return _matching(self.torch_from_numpy, _RUNNER_FRAMES)

    def runner_runs(self) -> List[Tuple[str, str]]:
        return _matching(self.runs, _RUNNER_FRAMES)

    def summary(self) -> Dict[str, Any]:
        """Compact, JSON-friendly per-block traffic summary."""
        return {
            "runs": len(self.runs),
            "runner_runs": len(self.runner_runs()),
            "readbacks": len(self.readbacks),
            "runner_readbacks": len(self.runner_readbacks()),
            "uploads": len(self.uploads),
            "preparations": len(self.preparations),
            "torch_from_host": len(self.torch_from_numpy),
            "build_cached": len(self.builds),
            "executable_lookups": len(self.lookups),
            "decodes": len(self.decodes),
        }


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def forest_with_const(
    pop: int, rows: int, *, input_len: int = 3, seed: int = 0, const_rows: int = 1
) -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    """A random population plus ``const_rows`` constant-only (never-compiled) trees.

    A constant tree is folded at decode time and never compiled, so it is the
    case that pins the runners' "no executable" branch (``None`` slot in the
    specialized runner's ``.run()``, ``.constant_rows``).
    """
    forest, inputs, labels = fixture(pop, rows, input_len=input_len, seed=seed)
    if const_rows:
        const_tree = pack([[("const", 2.5)]], forest.max_tree_len, input_len)[0]
        for index in range(min(int(const_rows), int(pop))):
            forest[index] = const_tree
    return forest, inputs, labels


def broken_forest(forest: Forest) -> Forest:
    """A copy of ``forest`` whose first row is malformed (a tree longer than the row)."""
    import copy

    broken = copy.copy(forest)
    sizes = broken.batch_subtree_size.clone()
    sizes[0, 0] = sizes.shape[1] + 1  # declared length beyond the row
    return Forest(
        forest.input_len,
        forest.output_len,
        broken.batch_node_value,
        broken.batch_node_type,
        sizes,
    )


def reference_rows(forest: Forest, x: Any, *, per_row: bool, backend: str) -> torch.Tensor:
    """The specialized engine's rows for this fixture (the numeric reference)."""
    population = E.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    return E.evaluate(population, x, per_row=per_row, backend=backend)


def raw_to_host(raw: Any, backend: str) -> np.ndarray:
    """Host copy of one raw evoxc result -- done by the CHECK, never by the runner."""
    array = raw.numpy() if backend == "numpy" else raw.to(E._CPU).numpy()
    return np.asarray(array, dtype=np.float32)


def median_run_ms(runner: Any, repeats: int) -> float:
    """Warm median wall time of one ``.run()`` (stream-synced when CUDA is visible)."""
    sync = torch.cuda.is_available()
    runner.run()
    samples: List[float] = []
    for _ in range(max(1, int(repeats))):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        runner.run()
        if sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(statistics.median(samples))


def _cache_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: after[key] - before[key] for key in after if after[key] != before[key]}


def _detail_counts(records: Sequence[Any]) -> str:
    counts: Dict[str, int] = {}
    for record in records:
        key = str(tuple(record[2:])) if len(record) > 2 else ""
        counts[key] = counts.get(key, 0) + 1
    return str(counts)


def _raises(exc_type: type, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[bool, str]:
    """``(raised exactly exc_type, message)``."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return True, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # wrong type: surfaced as a mismatch
        return False, f"{type(exc).__name__}: {exc}"
    return False, "no exception"


def _host_x(inputs: Any) -> np.ndarray:
    return np.ascontiguousarray(inputs.detach().cpu().numpy(), dtype=np.float32)


def control_dense_reference(forest: Forest, inputs: Any, args: Any, report: Report) -> None:
    """Positive control: the counters DO see a readback-owning dense consumer.

    ``runner_readbacks() == 0`` is only evidence if the same counters fire for a
    call that reads its result back (``run_plan``) and for a FRESH seam call that
    re-prepares / re-uploads the payloads (no injected memo).  Without this
    control a filter that silently stopped matching would look like a pass --
    exactly the failure mode device_resident_check.py's ``seam`` stage guards.
    """
    rows = int(inputs.shape[0])
    plan = D.plan_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        rows,
        per_row=False,
        backend=args.backend,
    )
    executable = D.build_dense(plan, backend=args.backend)
    backend_name, device, _ = E._resolve_backend(args.backend, args.gpu)
    x_host = _host_x(inputs)
    with Traffic() as control:
        D.run_plan(plan, executable, x_host, backend_name=backend_name, device=device)
    seen = control.reference_readbacks()
    report.add(
        "control: run_plan's per-chunk readbacks ARE visible to the counters",
        len(seen) == plan.p_chunks * plan.n_chunks and len(control.runs) == 1,
        f"{len(seen)} readbacks from {[record[:2] for record in seen]}, "
        f"{len(control.runs)} evoxc.run calls",
    )
    with Traffic() as fresh:
        chunks = list(
            D._iter_plan_chunks(
                plan,
                executable,
                x_host,
                backend_name=backend_name,
                device=device,
                device_resident=backend_name != "numpy",
            )
        )
    staged = fresh.uploads if backend_name != "numpy" else fresh.preparations
    report.add(
        "control: a FRESH seam call re-stages the payloads (counters live)",
        len(chunks) == 1
        and len(staged) == _DENSE_COLUMNS + 1
        and len(fresh.runner_readbacks()) == 0,
        f"{len(staged)} payload staging calls on {backend_name} "
        f"(expected {_DENSE_COLUMNS + 1}), {len(fresh.runner_readbacks())} readbacks",
    )


def control_specialized_reference(
    forest: Forest, inputs: Any, args: Any, report: Report
) -> None:
    """Positive control: ``evaluate``'s decode + per-tree readbacks ARE counted."""
    with Traffic() as control:
        population = E.decode_population(
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
            forest.input_len,
            forest.output_len,
        )
        E.evaluate(population, inputs, per_row=False, backend=args.backend)
    seen = control.reference_readbacks()
    report.add(
        "control: evaluate()'s decode + per-tree readbacks ARE visible",
        len(control.decodes) == 1 and len(seen) > 0 and len(control.runs) > 0,
        f"{len(control.decodes)} decode(s), {len(control.runs)} evoxc.run(s), "
        f"{len(seen)} readbacks from {sorted({record[0] for record in seen})}",
    )


# ---------------------------------------------------------------------------
# stage: dense runner
# ---------------------------------------------------------------------------


def stage_dense(args: Any, report: Report, results: Dict[str, Any]) -> None:
    print("== stage dense: build_dense_runner ==")
    builder = getattr(D, "build_dense_runner", None)
    if builder is None:
        report.add(
            "dense_engine.build_dense_runner exists",
            False,
            "not implemented: the dense benchmark runner API is missing",
        )
        return
    report.add("dense_engine.build_dense_runner exists", True, "")

    pop, rows, input_len = args.pop, args.rows, args.input_len
    forest, inputs, _ = forest_with_const(
        pop, rows, input_len=input_len, seed=7, const_rows=1
    )
    set_mode("specialized", backend=args.backend_env)

    # -- build phase: ONE compile, payloads staged once, one warm-up run ----
    E.clear_cache()
    before = E.cache_stats()
    start = time.perf_counter()
    with Traffic() as build:
        runner = builder(forest, inputs, backend=args.backend)
    build_ms = (time.perf_counter() - start) * 1e3
    after = E.cache_stats()
    plan = runner.plan
    report.add(
        "dense build: exactly ONE compile (one dense cache entry)",
        after["misses"] - before["misses"] == 1 and after["entries"] - before["entries"] == 1,
        f"misses {before['misses']} -> {after['misses']}, entries {before['entries']} -> "
        f"{after['entries']} for a population of {pop} distinct structures",
    )
    report.add(
        "dense build: never touches the per-tree executable memo",
        after["memo_misses"] == before["memo_misses"],
        f"memo_misses {before['memo_misses']} -> {after['memo_misses']}",
    )
    report.add(
        "dense build: single-chunk plan (the runner's contract)",
        plan.p_chunks == 1 and plan.n_chunks == 1,
        f"P_chunk={plan.p_chunk} N_chunk={plan.n_chunk} "
        f"({plan.p_chunks}x{plan.n_chunks} chunks)",
    )
    if args.backend == "numpy":
        columns = [r for r in build.preparations if r[2] and r[2][-1] == plan.trace_len]
        blocks = [r for r in build.preparations if r[2] and r[2][-1] == input_len]
        payload_detail = (
            f"numpy backend: {len(columns)} column preparations (last dim="
            f"{plan.trace_len}), {len(blocks)} x preparations (last dim={input_len}); "
            f"{len(build.uploads)} upload calls (a device backend uploads)"
        )
    else:
        columns = [r for r in build.uploads if r[2] == (plan.trace_len, plan.p_chunk)]
        blocks = [r for r in build.uploads if r[2] == (plan.n_chunk, input_len)]
        payload_detail = (
            f"uploads {_detail_counts(build.uploads)} for a "
            f"{plan.trace_len}x{plan.p_chunk} column block and a "
            f"{plan.n_chunk}x{input_len} x block"
        )
    report.add(
        "dense build: stages the 7 node columns + x exactly once",
        len(columns) == _DENSE_COLUMNS and len(blocks) == 1,
        f"columns={len(columns)} (expected {_DENSE_COLUMNS}), x={len(blocks)} "
        f"(expected 1) -- {payload_detail}",
    )
    report.add(
        "dense build: one warm-up run and zero readbacks",
        len(build.runs) == 1 and len(build.runner_readbacks()) == 0,
        f"{len(build.runs)} evoxc.run calls, {len(build.runner_readbacks())} readbacks "
        f"from runner/seam frames",
    )

    # -- run phase: ONE evoxc.run, zero crossings, no re-key ---------------
    with Traffic() as probe:
        raw = runner.run()
    staged_before = {key: id(value[0]) for key, value in runner.staged.items()}
    probe_ok = (
        len(probe.runs) == 1
        and len(probe.runner_readbacks()) == 0
        and len(probe.uploads) == 0
        and len(probe.preparations) == 0
        and len(probe.torch_from_numpy) == 0
        and len(probe.builds) == 0
        and len(probe.lookups) == 0
        and len(probe.decodes) == 0
    )
    report.add(
        "dense run(): exactly ONE evoxc.run, no readback / upload / preparation /",
        probe_ok,
        f"{probe.summary()}",
    )
    report.add(
        "dense run(): no torch conversion and no re-plan / re-key",
        len(probe.torch_from_numpy) == 0
        and len(probe.builds) == 0
        and len(probe.lookups) == 0,
        f"torch_from_numpy={len(probe.torch_from_numpy)} build_cached={len(probe.builds)} "
        f"_executable={len(probe.lookups)} for a runner built with plan#"
        f"{id(plan) & 0xFFFF}",
    )
    first = raw_to_host(raw, args.backend).reshape(plan.pop_size, plan.n_rows)

    # -- repeatability -----------------------------------------------------
    repeats = max(1, int(args.repeats))
    same = True
    zero_traffic = True
    detail = ""
    for index in range(repeats):
        with Traffic() as again:
            repeat_raw = runner.run()
        zero_traffic = zero_traffic and (
            len(again.runs) == 1
            and len(again.runner_readbacks()) == 0
            and len(again.uploads) == 0
            and len(again.preparations) == 0
            and len(again.torch_from_numpy) == 0
        )
        equal, note = _nan_safe_equal(
            first, raw_to_host(repeat_raw, args.backend).reshape(plan.pop_size, plan.n_rows)
        )
        if not equal:
            same, detail = False, f"call {index + 1}: {note}"
            break
    report.add(
        f"dense run(): {repeats} further calls are bit-identical and readback-free",
        same and zero_traffic,
        detail or f"{repeats} calls, zero host crossings each",
    )
    after_runs = E.cache_stats()
    report.add(
        "dense run(): ZERO new cache activity after the build",
        _cache_delta(after, after_runs) == {},
        f"cache deltas {_cache_delta(after, after_runs) or 'none'}",
    )
    report.add(
        "dense run(): the staged payload memo is untouched",
        staged_before == {key: id(value[0]) for key, value in runner.staged.items()},
        f"{len(runner.staged)} staged payloads, all re-used by reference",
    )

    # -- positive controls: the counters DO fire on readback-owning paths ---
    control_dense_reference(forest, inputs, args, report)

    # -- numerics ----------------------------------------------------------
    reference = reference_rows(forest, inputs, per_row=False, backend=args.backend)
    compare(
        "dense run() rows == evoxc_engine.evaluate(per_row=False)",
        reference[:, :, 0],
        first,
        report,
    )
    if args.backend == "numpy":
        reference_cpu = reference.detach().cpu().numpy()
        compare(
            "dense run() rows == Forest.batch_forward",
            reference_cpu[:, :, 0],
            forest.batch_forward(inputs).detach().cpu().numpy()[:, :, 0],
            report,
        )

    # -- per_row=True ------------------------------------------------------
    per_row_forest, per_row_x, _ = forest_with_const(
        pop, pop, input_len=input_len, seed=11, const_rows=1
    )
    set_mode("specialized", backend=args.backend_env)
    with Traffic() as build_per_row:
        per_row_runner = builder(per_row_forest, per_row_x, per_row=True, backend=args.backend)
    with Traffic() as probe_per_row:
        per_row_raw = per_row_runner.run()
    per_row_host = raw_to_host(per_row_raw, args.backend).reshape(pop, 1)
    report.add(
        "dense per_row run(): ONE evoxc.run on a single-chunk plan, no host crossings",
        len(probe_per_row.runs) == 1
        and len(probe_per_row.runner_readbacks()) == 0
        and len(probe_per_row.uploads) == 0
        and len(probe_per_row.torch_from_numpy) == 0
        and per_row_runner.plan.p_chunks == 1
        and per_row_runner.plan.n_chunks == 1,
        f"build {len(build_per_row.runs)} run(s); {probe_per_row.summary()}",
    )
    compare(
        "dense per_row run() == evoxc_engine.evaluate(per_row=True)",
        reference_rows(per_row_forest, per_row_x, per_row=True, backend=args.backend),
        per_row_host,
        report,
    )
    if args.backend == "numpy":
        compare(
            "dense per_row run() == Forest.forward",
            per_row_forest.forward(per_row_x).detach().cpu().numpy(),
            per_row_host,
            report,
        )
    set_mode("auto", backend="numpy")

    # -- timings -----------------------------------------------------------
    run_ms = median_run_ms(runner, repeats)
    print(
        f"[time] dense  P={pop} N={rows} L={plan.trace_len} backend={args.backend}: "
        f"build={build_ms:.2f} ms, .run() median={run_ms:.3f} ms "
        f"({repeats} calls, {plan.p_chunks}x{plan.n_chunks} chunk)"
    )
    results["dense"] = {
        "pop": pop,
        "rows": rows,
        "trace_len": int(plan.trace_len),
        "backend": args.backend,
        "build_ms": build_ms,
        "run_median_ms": run_ms,
        "run_calls": 1,
        "traffic": probe.summary(),
        "build_traffic": build.summary(),
    }


# ---------------------------------------------------------------------------
# stage: specialized runner
# ---------------------------------------------------------------------------


def _assemble_specialized(runner: Any, raws: Sequence[Any], backend: str) -> np.ndarray:
    """Host ``(P, N, out)`` / ``(P, out)`` rows from the runner's raw results.

    The constant trees have no evoxc result (``None``): their rows come from the
    broadcast ``.constant_rows`` the build staged.
    """
    pop, n_rows, out_len = runner.pop_size, runner.n_rows, runner.output_len
    stitched = np.empty((pop, n_rows, out_len), dtype=np.float32)
    for index, raw in enumerate(raws):
        if raw is None:
            stitched[index] = runner.constant_rows[index]
        else:
            stitched[index] = raw_to_host(raw, backend).reshape(n_rows, out_len)
    return stitched[:, 0, :] if runner.per_row else stitched


def stage_specialized(args: Any, report: Report, results: Dict[str, Any]) -> None:
    print("== stage specialized: build_specialized_runner ==")
    builder = getattr(E, "build_specialized_runner", None)
    if builder is None:
        report.add(
            "evoxc_engine.build_specialized_runner exists",
            False,
            "not implemented: the specialized benchmark runner API is missing",
        )
        return
    report.add("evoxc_engine.build_specialized_runner exists", True, "")

    pop, rows, input_len = args.pop, args.rows, args.input_len
    const_rows = 1
    forest, inputs, _ = forest_with_const(
        pop, rows, input_len=input_len, seed=13, const_rows=const_rows
    )
    set_mode("specialized", backend=args.backend_env)

    # -- build phase: decode once, resolve every executable, stage inputs --
    E.clear_cache()
    before = E.cache_stats()
    start = time.perf_counter()
    with Traffic() as build:
        runner = builder(forest, inputs, backend=args.backend)
    build_ms = (time.perf_counter() - start) * 1e3
    after = E.cache_stats()
    # The fixture injects one constant tree; a random population can contain more
    # (a lone CONST node, a fully foldable constant subtree), and that is exactly
    # the case the runner has to handle -- so count them from the decode.
    n_const = sum(1 for tree in runner.population.trees if tree.const_out is not None)
    n_nonconst = pop - n_const
    programs = [tree.program for tree in runner.population.trees if tree.program is not None]
    distinct = len(set(programs))
    report.add(
        "specialized build: decodes the population exactly once",
        len(build.decodes) == 1,
        f"{len(build.decodes)} decode_population call(s)",
    )
    report.add(
        "specialized build: resolves EVERY per-tree executable once",
        len(build.lookups) == n_nonconst,
        f"{len(build.lookups)} _executable lookups for {n_nonconst} non-constant "
        f"trees ({n_const} constant trees folded at decode, never compiled)",
    )
    report.add(
        "specialized build: one compile per DISTINCT program",
        len(build.builds) == distinct
        and after["misses"] - before["misses"] == distinct
        and after["memo_misses"] - before["memo_misses"] == distinct,
        f"{len(build.builds)} build_cached calls, {distinct} distinct programs, "
        f"cache misses {before['misses']} -> {after['misses']}",
    )
    report.add(
        "specialized build: never runs a graph, never reads a result back",
        len(build.runs) == 0 and len(build.runner_readbacks()) == 0,
        f"{len(build.runs)} evoxc.run calls, {len(build.runner_readbacks())} readbacks",
    )
    expected_uploads = 1 if args.backend != "numpy" else 0
    report.add(
        "specialized build: stages the input block once (per_row=False)",
        len(build.uploads) == expected_uploads and len(build.preparations) == 0,
        f"uploads={len(build.uploads)} (expected {expected_uploads}), "
        f"preparations={len(build.preparations)} (this path never pads)",
    )

    # -- run phase: n_nonconst evoxc.run calls, nothing else ---------------
    with Traffic() as probe:
        raws = runner.run()
    ok = (
        len(raws) == pop
        and sum(1 for raw in raws if raw is None) == n_const
        and len(probe.runs) == n_nonconst
        and len(probe.runner_readbacks()) == 0
        and len(probe.uploads) == 0
        and len(probe.preparations) == 0
        and len(probe.torch_from_numpy) == 0
        and len(probe.builds) == 0
        and len(probe.lookups) == 0
        and len(probe.decodes) == 0
    )
    report.add(
        "specialized run(): exactly n_nonconst evoxc.run calls, no host crossings",
        ok,
        f"results={len(raws)} (None slots="
        f"{sum(1 for raw in raws if raw is None)}, expected {n_const}); "
        f"{probe.summary()}",
    )
    first = _assemble_specialized(runner, raws, args.backend)

    repeats = max(1, int(args.repeats))
    same = True
    zero_traffic = True
    detail = ""
    for index in range(repeats):
        with Traffic() as again:
            repeat_raws = runner.run()
        zero_traffic = zero_traffic and (
            len(again.runs) == n_nonconst
            and len(again.runner_readbacks()) == 0
            and len(again.uploads) == 0
            and len(again.decodes) == 0
            and len(again.lookups) == 0
        )
        equal, note = _nan_safe_equal(
            first, _assemble_specialized(runner, repeat_raws, args.backend)
        )
        if not equal:
            same, detail = False, f"call {index + 1}: {note}"
            break
    report.add(
        f"specialized run(): {repeats} further calls are bit-identical and readback-free",
        same and zero_traffic,
        detail or f"{repeats} calls, exactly {n_nonconst} evoxc.run each",
    )
    after_runs = E.cache_stats()
    report.add(
        "specialized run(): ZERO new cache activity after the build",
        _cache_delta(after, after_runs) == {},
        f"cache deltas {_cache_delta(after, after_runs) or 'none'}",
    )

    # -- positive control: the counters DO fire on decode + readback -------
    control_specialized_reference(forest, inputs, args, report)

    # -- numerics ----------------------------------------------------------
    compare(
        "specialized run() rows == evoxc_engine.evaluate(per_row=False)",
        reference_rows(forest, inputs, per_row=False, backend=args.backend),
        first,
        report,
    )
    if args.backend == "numpy":
        compare(
            "specialized run() rows == Forest.batch_forward",
            forest.batch_forward(inputs).detach().cpu().numpy(),
            first,
            report,
        )

    # -- per_row=True: one staged row per tree -----------------------------
    per_row_forest, per_row_x, _ = forest_with_const(
        pop, pop, input_len=input_len, seed=17, const_rows=const_rows
    )
    with Traffic() as build_per_row:
        per_row_runner = builder(per_row_forest, per_row_x, per_row=True, backend=args.backend)
    with Traffic() as probe_per_row:
        per_row_raws = per_row_runner.run()
    per_row_nonconst = per_row_runner.invocations
    report.add(
        "specialized build(per_row=True): one staged row per tree",
        len(build_per_row.uploads) == (
            per_row_nonconst if args.backend != "numpy" else 0
        ),
        f"uploads={len(build_per_row.uploads)} for {per_row_nonconst} "
        "non-constant trees",
    )
    report.add(
        "specialized run(per_row=True): n_nonconst runs, no host crossings",
        len(probe_per_row.runs) == per_row_nonconst
        and len(probe_per_row.runner_readbacks()) == 0
        and len(probe_per_row.uploads) == 0,
        f"{probe_per_row.summary()}",
    )
    compare(
        "specialized per_row run() == evoxc_engine.evaluate(per_row=True)",
        reference_rows(per_row_forest, per_row_x, per_row=True, backend=args.backend),
        _assemble_specialized(per_row_runner, per_row_raws, args.backend),
        report,
    )
    if args.backend == "numpy":
        compare(
            "specialized per_row run() == Forest.forward",
            per_row_forest.forward(per_row_x).detach().cpu().numpy(),
            _assemble_specialized(per_row_runner, per_row_raws, args.backend),
            report,
        )

    # -- multi-output (the dense path's DenseUnsupported case) -------------
    multi = multi_output_forest()
    multi_x = torch.tensor([[2.0], [5.0]], dtype=torch.float32, device="cuda")
    with Traffic() as build_multi:
        multi_runner = builder(multi, multi_x, backend=args.backend)
    with Traffic() as probe_multi:
        multi_raws = multi_runner.run()
    report.add(
        "specialized runner handles a multi-output population",
        len(probe_multi.runs) == len(multi_raws) == 2
        and all(raw is not None for raw in multi_raws)
        and len(build_multi.decodes) == 1,
        f"decodes={len(build_multi.decodes)}, runs={len(probe_multi.runs)}, "
        f"output_len={multi_runner.output_len}",
    )
    compare(
        "specialized multi-output rows == evoxc_engine.evaluate(per_row=False)",
        reference_rows(multi, multi_x, per_row=False, backend=args.backend),
        _assemble_specialized(multi_runner, multi_raws, args.backend),
        report,
    )

    set_mode("auto", backend="numpy")

    # -- timings -----------------------------------------------------------
    run_ms = median_run_ms(runner, repeats)
    print(
        f"[time] specialized P={pop} N={rows} backend={args.backend}: "
        f"build={build_ms:.2f} ms ({distinct} compiles), .run() median={run_ms:.3f} ms "
        f"({repeats} calls, {n_nonconst} trees/call)"
    )
    results["specialized"] = {
        "pop": pop,
        "rows": rows,
        "backend": args.backend,
        "build_ms": build_ms,
        "distinct_programs": distinct,
        "non_const_trees": n_nonconst,
        "run_median_ms": run_ms,
        "run_calls": n_nonconst,
        "traffic": probe.summary(),
        "build_traffic": build.summary(),
    }


# ---------------------------------------------------------------------------
# stage: edges (build-time error contracts + policy independence)
# ---------------------------------------------------------------------------


def stage_edges(args: Any, report: Report, results: Dict[str, Any]) -> None:
    print("== stage edges: build-time contracts ==")
    dense_builder = getattr(D, "build_dense_runner", None)
    specialized_builder = getattr(E, "build_specialized_runner", None)
    if dense_builder is None or specialized_builder is None:
        report.add(
            "both runner builders exist",
            False,
            "build_dense_runner / build_specialized_runner are not both implemented",
        )
        return
    report.add("both runner builders exist", True, "")

    pop, input_len = 4, 3
    forest, inputs, _ = forest_with_const(
        pop, 8, input_len=input_len, seed=23, const_rows=1
    )
    multi = multi_output_forest()
    multi_x = torch.tensor([[2.0], [5.0]], dtype=torch.float32, device="cuda")

    ok, note = _raises(
        D.DenseUnsupported, dense_builder, multi, multi_x, backend=args.backend
    )
    report.add(
        "dense builder refuses a multi-output population (DenseUnsupported)",
        ok,
        note,
    )
    ok, note = _raises(
        D.DenseUnsupported,
        dense_builder,
        forest,
        inputs,
        backend=args.backend,
        budget_mb=1e-6,
    )
    report.add(
        "dense builder refuses a CHUNKED plan (DenseUnsupported)",
        ok,
        note,
    )
    ok, note = _raises(
        ValueError,
        dense_builder,
        forest,
        torch.randn(inputs.shape[0], input_len + 1, device="cuda"),
        backend=args.backend,
    )
    report.add("dense builder rejects a wrong-columns inputs (ValueError)", ok, note)
    ok, note = _raises(
        ValueError, dense_builder, forest, inputs, per_row=True, backend=args.backend
    )
    report.add(
        "dense builder rejects per_row with rows != pop_size (ValueError)", ok, note
    )
    ok, note = _raises(TypeError, dense_builder, object(), inputs, backend=args.backend)
    report.add("dense builder rejects a non-Forest-like container (TypeError)", ok, note)

    ok, note = _raises(
        ValueError,
        specialized_builder,
        forest,
        torch.randn(inputs.shape[0], input_len + 1, device="cuda"),
        backend=args.backend,
    )
    report.add(
        "specialized builder rejects a wrong-columns inputs (ValueError)", ok, note
    )
    ok, note = _raises(
        ValueError,
        specialized_builder,
        forest,
        inputs,
        per_row=True,
        backend=args.backend,
    )
    report.add(
        "specialized builder rejects per_row with rows != pop_size (ValueError)",
        ok,
        note,
    )
    ok, note = _raises(
        TypeError, specialized_builder, object(), inputs, backend=args.backend
    )
    report.add(
        "specialized builder rejects a non-Forest-like container (TypeError)", ok, note
    )
    ok, note = _raises(
        ValueError, specialized_builder, broken_forest(forest), inputs, backend=args.backend
    )
    report.add(
        "specialized builder rejects a malformed tree (decode ValueError)", ok, note
    )

    # -- policy independence: the runners never consult EVOGP_EVAL_MODE ----
    small, small_x, _ = forest_with_const(pop, 4, input_len=input_len, seed=29, const_rows=1)
    set_mode("specialized", backend=args.backend_env)
    try:
        dense_runner = dense_builder(small, small_x, backend=args.backend)
        dense_is_dense = isinstance(dense_runner.plan, D.DensePlan)
        dense_note = f"EVOGP_EVAL_MODE=specialized, plan={type(dense_runner.plan).__name__}"
    except Exception as exc:  # a fallback would raise/return a non-dense runner
        dense_is_dense = False
        dense_note = f"{type(exc).__name__}: {exc}"
    set_mode("dense", backend=args.backend_env)
    try:
        specialized_runner = specialized_builder(small, small_x, backend=args.backend)
        specialized_is_specialized = (
            isinstance(specialized_runner.population, E.Population)
            and any(
                executable is not None for executable in specialized_runner.executables
            )
        )
        specialized_note = (
            f"EVOGP_EVAL_MODE=dense, population="
            f"{type(specialized_runner.population).__name__}"
        )
    except Exception as exc:
        specialized_is_specialized = False
        specialized_note = f"{type(exc).__name__}: {exc}"
    report.add(
        "both runners ignore EVOGP_EVAL_MODE (explicit route selection)",
        dense_is_dense and specialized_is_specialized,
        f"dense: {dense_note}; specialized: {specialized_note}",
    )
    # The builders must not touch the route switch themselves: neither the mode
    # nor the backend variable may change while they run.
    report.add(
        "the builders leave EVOGP_EVAL_MODE / EVOGP_EVOXC_BACKEND untouched",
        os.environ.get(D.EVAL_MODE_ENV) == "dense"
        and os.environ.get("EVOGP_EVOXC_BACKEND") == args.backend_env,
        f"EVOGP_EVAL_MODE={os.environ.get(D.EVAL_MODE_ENV)!r} "
        f"EVOGP_EVOXC_BACKEND={os.environ.get('EVOGP_EVOXC_BACKEND')!r}",
    )
    set_mode("auto", backend="numpy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        default="auto",
        help=(
            "comma list of stages to run: dense,specialized,edges; 'auto' = all "
            "three on numpy, dense+specialized on a device backend"
        ),
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="engine backend for the runners ('numpy' or 'iree')",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="visible CUDA index for the iree backend (EVOGP_EVOXC_DEVICE)",
    )
    parser.add_argument("--pop", type=int, default=None, help="population size")
    parser.add_argument("--rows", type=int, default=None, help="data rows")
    parser.add_argument("--input-len", type=int, default=3, help="input columns")
    parser.add_argument(
        "--repeats", type=int, default=7, help="warm .run() repeats for repeatability/timing"
    )
    parser.add_argument("--json", default=None, help="write the timings/traffic JSON here")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stages = (
        list(_STAGES)
        if args.stage == "auto"
        else [name.strip() for name in args.stage.split(",") if name.strip()]
    )
    iree = args.backend != "numpy"
    if args.stage == "auto" and iree:
        stages = ["dense", "specialized"]
    for name in stages:
        if name not in _STAGES:
            print(f"unknown stage {name!r}: expected one of {list(_STAGES)}")
            return 2
    # The engine backend is always NAMED for the builders (``set_mode`` writes it
    # explicitly so a forced ``dense`` leg cannot fall through to the device route
    # on a GPU box); ``_backend`` still treats ``"numpy"`` as the default (unset).
    args.backend_env = args.backend
    if args.pop is None:
        args.pop = 4 if iree else 8
    if args.rows is None:
        args.rows = 8 if iree else 37
    print(
        f"runner_check: stages={stages} backend={args.backend} gpu={args.gpu} "
        f"pop={args.pop} rows={args.rows} input_len={args.input_len} "
        f"repeats={args.repeats}"
    )

    report = Report()
    results: Dict[str, Any] = {"backend": args.backend, "pop": args.pop, "rows": args.rows}
    with _backend(args.backend_env, args.gpu):
        if "dense" in stages:
            stage_dense(args, report, results)
        if "specialized" in stages:
            stage_specialized(args, report, results)
        if "edges" in stages:
            stage_edges(args, report, results)

    passed = sum(1 for entry in report.entries if entry["ok"])
    total = len(report.entries)
    results["checks"] = report.entries
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        print(f"[json] {args.json}")
    if report.ok:
        print(f"== OK: {passed}/{total} checks passed ==")
        return 0
    print(f"== FAILED: {passed}/{total} checks passed ==")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # hard error: a crash is not a check failure
        import traceback

        traceback.print_exc()
        print(f"== HARD ERROR: {type(exc).__name__}: {exc} ==")
        sys.exit(2)
