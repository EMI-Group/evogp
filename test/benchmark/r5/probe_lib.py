#!/usr/bin/env python
"""``probe_lib.py`` -- shared runtime-instrumentation helpers for the R5 RCA probes.

R5 is analysis-only: nothing here edits ``src/evogp`` or the foreign ``etl``
checkout.  Every measurement is a **runtime monkeypatch** of the engine's own
call sites (the ``test/benchmark/stage_probe.py`` pattern, extended: exclusive
per-span accounting instead of residuals, and a scoped ``etl`` module proxy).

What this file exports (the stable surface later probes import)
--------------------------------------------------------------
``NOW``                            ``time.perf_counter`` (single clock source)
``Profiler``                       stack-based exclusive-accounting timer +
                                   per-call records (see below)
``Frame``                          one open span (internal, exposed for typing)
``patch_attr(target, name, ...)``  monkeypatch a module/class attribute with a
                                   profiler span (+ optional inline CUDA fence)
``EtlProxy``                       transparent stand-in for the ``etl`` module:
                                   delegates everything, times ``from_numpy`` /
                                   ``run`` in the *engine's* namespace only
``install_etl_proxy(dense_module, profiler)``   install ``EtlProxy``
``patch_etl_tensor_to(etl_module, profiler)``   time ``etl.core.Tensor.to`` by
                                   caller function name (payload upload vs the
                                   ``_to_host`` readback vs "other")
``install_dense_wraps(dense_module, forest_module, problem_module, profiler)``
                                   the Q1/Q5/Q7 wrap set (10 call sites)
``to_cuda`` / ``build_eval_case``  the ``bench_gp.py`` ``--task eval64`` wiring,
                                   copied (and cited) so a probe never imports
                                   the harness
``identity_stamp`` / ``env_knobs`` / ``gpu_sample`` / ``write_json`` / ``iso``
                                   record-stamp helpers (every R5 record carries
                                   evogp commit + etl file/HEAD + env + GPU)

Profiler model
--------------
``Profiler`` keeps a stack of open :class:`Frame` objects.  A frame books its
wall time and, on exit, adds that whole time to its parent's ``children_s``, so
``exclusive = total - children`` (the sum of every bucket's exclusive time is
the root span's total by construction -- that is what "reconciled" means here).
``book_external(label, seconds)`` books time spent *outside* any wrapper (a
``Tensor.to`` inside ``payload``, an inline CUDA fence) into a child bucket of
the currently open frame, so the enclosing span's exclusive time stays honest.

Recording is gated on ``Profiler.active``: only the region a probe explicitly
opens with :meth:`Profiler.begin_call` / :meth:`Profiler.end_call` is booked, so
process startup, fixture construction and the probe's own syscalls are never
charged to a component.  ``enabled=False`` keeps every wrapper installed but
makes it a single boolean test -- the uninstrumented control pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "NOW",
    "Frame",
    "Profiler",
    "EtlProxy",
    "patch_attr",
    "install_etl_proxy",
    "patch_etl_tensor_to",
    "install_dense_wraps",
    "to_cuda",
    "build_eval_case",
    "identity_stamp",
    "env_knobs",
    "gpu_sample",
    "write_json",
    "iso",
    "REPO_ROOT",
]

NOW = time.perf_counter
WRAP_FLAG = "_r5_probe_wrapped"
REPO_ROOT = os.getcwd()
#: the foreign ETL checkout r5 stamps into every record (read-only; override with
#: ``R5_ETL_REPO`` when a record deliberately measures another checkout).
ETL_REPO = os.environ.get("R5_ETL_REPO", "/mnt/local-ssd/bchuang/etl")

#: The Q1/Q5/Q7 bucket names, in report order (see ``install_dense_wraps``).
BUCKET_ORDER = (
    "outer_evaluate",
    "forest_sr_self",
    "sr_fitness_dense_self",
    "plan_dense_self",
    "prepare_dense_self",
    "as_numpy_d2h",
    "input_to_numpy_self",
    "policy_should_use_dense",
    "build_dense_build_cached",
    "run_loss_plan_self",
    "prep_column",
    "etl_from_numpy",
    "etl_tensor_to_payload",
    "etl_tensor_to_to_host",
    "etl_tensor_to_other",
    "etl_run",
    "to_host_self",
)


# ---------------------------------------------------------------------------
# profiler
# ---------------------------------------------------------------------------


class Frame:
    """One open span: its bucket label, start timestamp and children time."""

    __slots__ = ("name", "t0", "children_s")

    def __init__(self, name: str, t0: float):
        self.name = name
        self.t0 = t0
        self.children_s = 0.0


class Profiler:
    """Exclusive-time profiler over nested spans, with per-call records.

    Attributes:
        enabled: False turns every wrapper into a pass-through (control pass).
        active: True only between :meth:`begin_call` and :meth:`end_call`.
        buckets: ``name -> {"count", "total_s", "self_s"}`` over all calls.
        calls: one record per measured call (``index``, ``phase``, ``label``,
            ``outer_s`` = the root span's own wall, ``buckets`` = per-call
            exclusive/total/count, ``extras`` = caller histograms).
        callers: ``name -> {caller_function_name: count}`` for externally booked
            time (e.g. which function called ``etl.core.Tensor.to``).
        notes: free-form strings the probe adds (mis-wrap warnings etc.).
    """

    MAX_DEPTH = 64

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.active = False
        self.phase = "none"
        self.label: Optional[str] = None
        self.index: Optional[int] = None
        self.buckets: Dict[str, Dict[str, float]] = {}
        self.calls: List[Dict[str, Any]] = []
        self.callers: Dict[str, Dict[str, int]] = {}
        self.notes: List[str] = []
        #: probe-owned switches the wrappers consult per call (``sync_region``).
        self.flags: Dict[str, Any] = {}
        self._stack: List[Frame] = []
        self._cur: Optional[Dict[str, Any]] = None
        self.depth_overflow = 0
        self.install_status: Dict[str, Any] = {"wrapped": [], "missing": []}
    # -- booking ------------------------------------------------------------
    def _book(self, name: str, total_s: float, self_s: float, dtype: str) -> None:
        entry = self.buckets.get(name)
        if entry is None:
            entry = self.buckets[name] = {"count": 0, "total_s": 0.0, "self_s": 0.0}
        entry["count"] += 1
        entry["total_s"] += total_s
        entry["self_s"] += self_s
        cur = self._cur
        if cur is not None:
            per = cur["buckets"].setdefault(
                name, {"count": 0, "total_s": 0.0, "self_s": 0.0}
            )
            per["count"] += 1
            per["total_s"] += total_s
            per["self_s"] += self_s
        if dtype:  # bucket provenance histogram (e.g. "external:...": caller -> count)
            hist = self.callers.setdefault(dtype, {})
            hist[name] = hist.get(name, 0) + 1
    def book_external(self, name: str, seconds: float, caller: Optional[str] = None) -> None:
        """Book leaf time spent outside any wrapper, attributed to the open frame.

        Also adds the time to the open frame's ``children_s`` so the enclosing
        span does not double count it in its own exclusive time.
        """
        if not self.active:
            return
        if caller is not None:
            hist = self.callers.setdefault("external:" + name, {})
            hist[caller] = hist.get(caller, 0) + 1
        self._book(name, seconds, seconds, "external:" + name)
        if self._stack:
            self._stack[-1].children_s += seconds

    def _close(self, frame: Frame) -> None:
        total = NOW() - frame.t0
        self._book(frame.name, total, total - frame.children_s, "")
        if self._stack:
            self._stack[-1].children_s += total

    @contextmanager
    def span(self, name: str, sync_after: Optional[str] = None):
        """Open a span; with ``sync_after``, fence CUDA inside it and book that."""
        if not (self.enabled and self.active) or len(self._stack) >= self.MAX_DEPTH:
            if self.enabled and self.active and len(self._stack) >= self.MAX_DEPTH:
                self.depth_overflow += 1
            yield
            return
        frame = Frame(name, NOW())
        self._stack.append(frame)
        try:
            yield
        finally:
            if sync_after is not None:
                import torch

                t0 = NOW()
                torch.cuda.synchronize()
                self.book_external(sync_after, NOW() - t0)
            self._stack.pop()
            self._close(frame)

    # -- call boundaries ----------------------------------------------------
    def begin_call(self, index: int, phase: str, label: Optional[str] = None) -> None:
        self.index = index
        self.phase = phase
        self.label = label
        self._cur = {
            "index": index,
            "phase": phase,
            "label": label,
            "buckets": {},
            "outer_s": 0.0,
        }
        self.active = True

    def end_call(self, outer_s: Optional[float] = None) -> Dict[str, Any]:
        self.active = False
        frame = self._stack.pop() if self._stack else None
        record = self._cur or {"index": None, "phase": self.phase, "buckets": {}}
        if frame is not None:  # a wrapper leaked a span: close it, do not lose it
            self._close(frame)
            self.notes.append(
                f"close-on-end: span {frame.name!r} was still open at end_call"
            )
        record["outer_s"] = float(outer_s) if outer_s is not None else 0.0
        self.calls.append(record)
        self._cur = None
        return record

    # -- reporting ----------------------------------------------------------
    def exclusive_ms(self, name: str) -> float:
        entry = self.buckets.get(name)
        return 0.0 if entry is None else entry["self_s"] * 1e3

    def counts(self) -> Dict[str, int]:
        return {name: int(entry["count"]) for name, entry in self.buckets.items()}

    def summary(self, phase: str = "warm", skip_index0: Optional[bool] = None) -> Dict[str, Dict[str, float]]:
        """``name -> {count, median_self_ms, mean_self_ms, total_self_ms}`` for one phase.

        ``skip_index0`` drops each phase's first call (the cold one); it defaults
        to True for the ``warm`` phase and False elsewhere.
        """
        if skip_index0 is None:
            skip_index0 = phase == "warm"
        out: Dict[str, Dict[str, float]] = {}
        warm = [
            c
            for c in self.calls
            if c["phase"] == phase and not (skip_index0 and c["index"] == 0)
        ]
        for name in self.buckets:
            per = [c["buckets"].get(name, {}).get("self_s", 0.0) * 1e3 for c in warm]
            count = (
                int(np.median([c["buckets"].get(name, {}).get("count", 0) for c in warm]))
                if warm
                else 0
            )
            out[name] = {
                "count": count,
                "median_self_ms": float(np.median(per)) if per else 0.0,
                "mean_self_ms": float(np.mean(per)) if per else 0.0,
                "total_self_ms": float(np.sum(per)),
            }
        return out
    def per_call_ms(self, phase: Optional[str] = None, key: str = "outer_s") -> List[float]:
        return [
            c[key] * 1e3
            for c in self.calls
            if phase is None or c["phase"] == phase
        ]

    def per_call_bucket_ms(
        self, name: str, phase: Optional[str] = None
    ) -> List[float]:
        return [
            c["buckets"].get(name, {}).get("self_s", 0.0) * 1e3
            for c in self.calls
            if phase is None or c["phase"] == phase
        ]


# ---------------------------------------------------------------------------
# monkeypatching
# ---------------------------------------------------------------------------


def patch_attr(
    target: Any,
    name: str,
    bucket: str,
    profiler: Profiler,
    *,
    sync_after: Optional[str] = None,
    on_return: Optional[Callable[[Any, Tuple[Any, ...], Dict[str, Any]], None]] = None,
    required: bool = True,
) -> bool:
    """Wrap ``target.name`` so every call books one ``bucket`` span.

    The wrapper is a pure pass-through (same return value, same exceptions).
    ``on_return(result, args, kwargs)`` may observe the result (never mutate it).
    ``sync_after`` may be a bucket label or a zero-arg callable returning one (or
    None to skip the fence for this call) -- that is how the region-sync control
    is switched on and off between phases.
    Returns True when the attribute was wrapped (idempotent), False when the
    attribute was missing -- a missing optional site is recorded in
    ``profiler.install_status["missing"]`` instead of crashing the probe.
    """
    original = getattr(target, name, None)
    if original is None or not callable(original):
        profiler.install_status["missing"].append(f"{getattr(target, '__name__', target)}.{name}")
        return False
    if getattr(original, WRAP_FLAG, False):
        return True  # already ours

    def wrapper(*args, **kwargs):
        if not (profiler.enabled and profiler.active):
            return original(*args, **kwargs)
        frame = Frame(bucket, NOW())
        if len(profiler._stack) < Profiler.MAX_DEPTH:
            profiler._stack.append(frame)
            own = True
        else:
            profiler.depth_overflow += 1
            own = False
        try:
            result = original(*args, **kwargs)
        finally:
            fence = sync_after() if callable(sync_after) else sync_after
            if fence is not None:
                import torch

                t0 = NOW()
                torch.cuda.synchronize()
                profiler.book_external(fence, NOW() - t0)
            if own:
                profiler._stack.pop()
                profiler._close(frame)
        if on_return is not None:
            try:
                on_return(result, args, kwargs)
            except Exception as exc:  # observation must never break a measured call
                profiler.notes.append(f"on_return({name}) failed: {exc!r}")
        return result

    try:
        wrapper.__name__ = getattr(original, "__name__", name)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        wrapper.__wrapped__ = original
        wrapper.__dict__[WRAP_FLAG] = True
    except (AttributeError, TypeError):
        pass
    try:
        setattr(target, name, wrapper)
    except Exception as exc:
        profiler.install_status["missing"].append(f"{name} (setattr: {exc!r})")
        return False
    profiler.install_status["wrapped"].append(bucket if bucket != name else name)
    return True


def install_dense_wraps(
    dense_module: Any,
    forest_module: Any,
    sr_class: Any,
    profiler: Profiler,
    *,
    sync_region: bool = False,
) -> Dict[str, Any]:
    """Install the Q1/Q5/Q7 wrap set on the dense ``problem.evaluate`` chain.

    Wrapped chain (every name is looked up in the module global the engine
    itself calls, so the patch is complete without editing ``src/``)::
        SymbolicRegression.evaluate                     outer_evaluate
          Forest.SR_fitness                             forest_sr_self
            dense_engine.should_use_dense               policy_should_use_dense
            dense_engine.sr_fitness_dense               sr_fitness_dense_self
              _input_to_numpy                           input_to_numpy_self
              _as_numpy (5 calls: x, labels, 3 node)    as_numpy_d2h
              plan_dense                                plan_dense_self
                prepare_dense                           prepare_dense_self
              build_dense -> etl.build_cached           build_dense_build_cached
              run_loss_plan                             run_loss_plan_self
                _prep_column (7 calls)                  prep_column
                etl.from_numpy (10 payloads)            etl_from_numpy
                <Tensor>.to(device) (payload upload)    etl_tensor_to_payload
                etl.run (1 chunk)                       etl_run
                _to_host -> <Tensor>.to(cpu)            etl_tensor_to_to_host

    ``sync_region=True`` adds a ``torch.cuda.synchronize()`` *inside* the span of
    every wrapped call (booked as ``fence_after:<bucket>``): the control pass that
    shows where a GPU tail is waited for.
    """
    fences: Dict[str, Optional[str]] = {}
    names = (
        ("outer_evaluate", sr_class, "evaluate"),
        ("forest_sr_self", forest_module.Forest, "SR_fitness"),
        ("policy_should_use_dense", dense_module, "should_use_dense"),
        ("sr_fitness_dense_self", dense_module, "sr_fitness_dense"),
        ("input_to_numpy_self", dense_module, "_input_to_numpy"),
        ("as_numpy_d2h", dense_module, "_as_numpy"),
        ("plan_dense_self", dense_module, "plan_dense"),
        ("prepare_dense_self", dense_module, "prepare_dense"),
        ("build_dense_build_cached", dense_module, "build_dense"),
        ("run_loss_plan_self", dense_module, "run_loss_plan"),
        ("prep_column", dense_module, "_prep_column"),
        ("to_host_self", dense_module, "_to_host"),
    )
    for bucket, target, attr in names:
        # The fence label is resolved per call from profiler.flags, so the
        # region-sync control can be switched on/off between phases in one process.
        def fence_label(bucket=bucket, sync_region=sync_region, profiler=profiler):
            if not sync_region or not profiler.flags.get("sync_region"):
                return None
            return f"fence_after:{bucket}"

        fences[bucket] = f"fence_after:{bucket}" if sync_region else None
        patch_attr(target, attr, bucket, profiler, sync_after=fence_label)

    return {
        "wrapped": list(profiler.install_status["wrapped"]),
        "missing": list(profiler.install_status["missing"]),
        "fences": fences,
    }


def install_plan_capture(dense_module: Any, sink: Dict[str, Any], profiler: Profiler) -> None:
    """Capture the first returned :class:`DensePlan` into ``sink`` (diagnostics).
    Installed *around* the ``plan_dense`` span wrapper, so the plan's own
    construction time is unaffected.
    """
    original = getattr(dense_module, "plan_dense")

    def wrapper(*args, **kwargs):
        plan = original(*args, **kwargs)
        try:
            if not sink:
                from evogp.tree import dense_engine as de

                sink.update(de.describe_plan(plan))
                sink["_spec_count"] = len(plan.specs)
                sink["_cfg"] = list(plan.cfg)
                sink["_variant"] = plan.variant
        except Exception as exc:
            profiler.notes.append(f"plan capture failed: {exc!r}")
        return plan

    wrapper.__name__ = "plan_dense"
    wrapper.__dict__[WRAP_FLAG] = True
    setattr(dense_module, "plan_dense", wrapper)


class EtlProxy:
    """A transparent stand-in for the ``etl`` module, installed per-namespace.

    Only the two calls the dense route makes through the module object are
    timed (``from_numpy`` = the host->ETL-tensor wrap, ``run`` = one compiled
    graph invocation); every other attribute is delegated untouched, so
    ``etl.TensorSpec`` / ``etl.build_cached`` / the graph ops inside
    ``@etl.defn`` bodies keep working through the proxy.
    """

    def __init__(self, real: Any, profiler: Profiler):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_prof", profiler)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_real"), name, value)

    def from_numpy(self, array: Any) -> Any:
        prof = object.__getattribute__(self, "_prof")
        with prof.span("etl_from_numpy"):
            return object.__getattribute__(self, "_real").from_numpy(array)

    def run(self, executable: Any, *args: Any, **kwargs: Any) -> Any:
        prof = object.__getattribute__(self, "_prof")
        fence = (
            "fence_after:etl_run"
            if (prof.enabled and prof.flags.get("sync_region"))
            else None
        )
        with prof.span("etl_run", sync_after=fence):
            return object.__getattribute__(self, "_real").run(executable, *args, **kwargs)

def install_etl_proxy(dense_module: Any, profiler: Profiler) -> EtlProxy:
    """Replace ``dense_module.etl`` (only that module's global) with an EtlProxy."""
    real = getattr(dense_module, "etl", None)
    if real is None:
        profiler.install_status["missing"].append("dense_engine.etl")
        raise RuntimeError("dense_engine has no `etl` module global to proxy")
    proxy = EtlProxy(real, profiler)
    setattr(dense_module, "etl", proxy)
    profiler.install_status["wrapped"].append("etl (module proxy: from_numpy/run)")
    return proxy


def patch_etl_tensor_to(
    etl_module: Any, profiler: Profiler, caller_labels: Optional[Dict[str, str]] = None
) -> Any:
    """Time ``etl.core.Tensor.to`` and bucket it by the caller function name.

    ``caller_labels`` maps a caller's function name to a bucket (default:
    ``payload`` -> the dense payload upload, ``_to_host`` -> the loss readback,
    anything else -> ``etl_tensor_to_other``).  Returns the patched class.
    """
    labels = {
        "payload": "etl_tensor_to_payload",
        "_to_host": "etl_tensor_to_to_host",
    }
    if caller_labels:
        labels.update(caller_labels)
    cls = type(etl_module.from_numpy(np.zeros(1, dtype=np.float32)))
    original = cls.to
    if getattr(original, WRAP_FLAG, False):
        return cls

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors etl's own method name
        if not profiler.active:
            return original(self, *args, **kwargs)
        try:
            caller = sys._getframe(1).f_code.co_name
        except Exception:
            caller = "<unknown>"
        bucket = labels.get(caller, "etl_tensor_to_other")
        t0 = NOW()
        result = original(self, *args, **kwargs)
        profiler.book_external(bucket, NOW() - t0, caller=caller)
        return result

    to.__name__ = "to"
    to.__dict__[WRAP_FLAG] = True
    to.__wrapped__ = original
    cls.to = to
    profiler.install_status["wrapped"].append("etl.core.Tensor.to (by caller)")
    return cls


# ---------------------------------------------------------------------------
# the bench_gp.py --task eval64 wiring (copied, with the harness lines cited)
# ---------------------------------------------------------------------------


def to_cuda(array: Any, dtype: Any):
    """``bench_gp.py:746-748`` -- writable contiguous torch tensor on cuda."""
    import torch

    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).to("cuda")


def build_eval_case(fixture: str):
    """``(problem, forest, x, labels)`` for ``--task eval64``.

    Byte-for-byte the harness wiring of ``bench_gp.py:768-806``
    (``build_eval_case``) minus the optional ``--data-rows`` tiling (R5's Q4/Q1
    cells do not tile): ``Forest(input_len, output_len, batch_node_value=...,
    batch_node_type=..., batch_subtree_size=...)`` rebuilt from the fixture's
    ``forest__*`` arrays, plus ``SymbolicRegression(datapoints=x, labels=labels)``
    -- so ``problem.evaluate(forest)`` lands on the same dense SR-fitness call the
    harness times (``bench_gp.py:1374-1381``).
    """
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest

    data = np.load(fixture)
    x = to_cuda(data["x"], np.float32)
    labels = to_cuda(data["labels"], np.float32)
    forest = Forest(
        input_len=int(x.shape[1]),
        output_len=int(labels.shape[1]),
        batch_node_value=to_cuda(data["forest__node_value"], np.float32),
        batch_node_type=to_cuda(data["forest__node_type"], np.int16),
        batch_subtree_size=to_cuda(data["forest__subtree_size"], np.int16),
    )
    problem = SymbolicRegression(datapoints=x, labels=labels)
    return problem, forest, x, labels


# ---------------------------------------------------------------------------
# record stamps
# ---------------------------------------------------------------------------


def iso(when: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(when))


def _run(cmd: List[str], cwd: Optional[str] = None) -> str:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60
        )
    except Exception as exc:
        return f"<{type(exc).__name__}>"
    out = (proc.stdout or proc.stderr or "").strip()
    return out or "<empty>"


def env_knobs() -> Dict[str, str]:
    """Every ``ETL_*`` / ``EVOGP_*`` knob visible to the process."""
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith("ETL_") or k.startswith("EVOGP_")
    }


def gpu_sample(index: Optional[str] = None) -> Dict[str, str]:
    """Utilisation + memory + co-tenant sample for the visible GPU."""
    index = index if index is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return {
        "index": index,
        "util_mem": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                index,
            ]
        ),
        "compute_apps": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
                "-i",
                index,
            ]
        ),
    }


def identity_stamp(etl_module: Any = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """evogp commit + etl file/HEAD + interpreter/versions + env + GPU sample."""
    import torch

    stamp: Dict[str, Any] = {
        "when": iso(),
        "cwd": REPO_ROOT,
        "evogp_commit": _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "evogp_file": os.path.join(REPO_ROOT, "src", "evogp", "__init__.py"),
        "etl_repo": ETL_REPO,
        "etl_repo_head": _run(["git", "rev-parse", "HEAD"], cwd=ETL_REPO),
        "etl_repo_status": _run(["git", "status", "--short"], cwd=ETL_REPO) or "<clean>",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "<n/a>"),
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_available": bool(torch.cuda.is_available()),
        "env_knobs": env_knobs(),
        "gpu": gpu_sample(),
    }
    if etl_module is not None:
        stamp["etl_file"] = getattr(etl_module, "__file__", "<n/a>")
        stamp["etl_version"] = getattr(etl_module, "__version__", "<n/a>")
    if extra:
        stamp.update(extra)
    return stamp


def write_json(path: str, payload: Any) -> None:
    """Atomic JSON write (tmp file + rename); the caller owns the dir."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)
