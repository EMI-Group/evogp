"""Measurement harness: integrated dense evaluator vs the specialized engine.

Runnable script, deliberately NOT a pytest module (no ``test_`` prefix, no pytest
import): it measures the dense one-graph evaluator
(:mod:`evogp.tree.dense_engine`, wired into the public API behind
``EVOGP_EVAL_MODE``) against the specialized per-structure engine
(:mod:`evogp.tree.etl_engine`) **through the public API only**
(``Forest.SR_fitness``, CUDA tensors), for a grid of ``pop_size`` x data rows, on
the ``numpy`` and ``iree`` backends.

What one case measures
----------------------
``plan`` (free)
    :func:`dense_engine.plan_dense` on the very population the run uses:
    ``trace_len``, realized chunk shape (``p_chunk``/``n_chunk``,
    ``p_chunks``/``n_chunks``) and the documented peak-memory model
    (:func:`dense_engine.dense_peak_bytes`).

``dense`` (public API)
    ``EVOGP_EVAL_MODE=dense`` + ``Forest.SR_fitness(inputs, labels)``.
    *cold* = one call after ``etl_engine.clear_cache()`` (that call pays the one
    dense compile); *warm* = median of ``--repeats`` calls after ``--warmup``
    warm-ups (all cache hits).  ``etl_engine.cache_stats()`` deltas confirm the
    cold call built exactly one executable.

``specialized`` (public API)
    ``EVOGP_EVAL_MODE=specialized`` + the same ``Forest.SR_fitness`` call.
    On ``numpy`` the *cold*/*warm* numbers are measured directly on the whole
    population (one compile per distinct tree structure is only ~10-20 ms
    there).  On ``iree`` one compile per tree is ~1 s, so a full-population cold
    run at ``pop_size >= 64`` is infeasible -- and the etl/iree binding segfaults
    after ~25 live executables per process -- so the harness measures a **sample**
    of single-tree populations (``--spec-sample``, cache cleared per tree) and
    models the population as

        warm_ms(P)    ~= decode_ms(P) + P * median(per-tree warm run ms)
        compile_ms(P) ~= #distinct programs * median(per-tree compile ms)

    plus (in ``--stage real``) one REAL small-population end-to-end call
    (``--spec-real-pop``, default 16) to validate that model.  The sample is
    measured once per ``N`` on the reference population (``--spec-sample-pop``,
    default the first ``--pops`` entry) because the per-tree cost does not
    depend on ``pop_size``.  On ``numpy`` the same model is stored next to the
    real numbers (``specialized.sample_model``) so its error is visible.

Correctness stamp (``verify``)
------------------------------
Every timed output is compared against a **numpy-specialized reference** built
with the public API on a tree subset (``--verify-rows`` trees; the whole
population when it is small): NaN masks and +-inf masks must match exactly,
finite values must agree within rtol/atol ``1e-4``.  A case whose timed output
is not verified is marked ``verified: false`` in the table.

Cost model / feasibility guards
-------------------------------
Dense cost grows with ``L * pop_size * rows`` (the traced graph carries ~264 ops
per node position) and the specialized warm cost with ``pop_size``, so corners
like ``pop 1000 x 10000 rows`` cannot always be run.  Before each case the
harness estimates its cost from the cases already measured in this process
(per-pop-size linear fit in ``rows``) and **skips** it above
``--max-config-seconds``; skipped corners are reported as clearly labelled
extrapolations (``source=fit(...)``) computed with the same fit, and the fit
itself is printed and stored in the JSON.  ``--repeats`` is chosen per case from
the estimate (3 / 2 / 1) so a slow corner is still timed at least once.
Dense host memory is guarded the same way (``--max-host-mb``) using the
*per-chunk* element count, because chunked execution is sequential: a corner that
the full plan cannot run can be measured for real by forcing chunking with a
small ``--budget-mb`` (the case record then holds the chunked plan and the
previous plan under ``plans_previous``).

Commands (repo root as CWD; the venv has torch, ``python`` on PATH does not)
---------------------------------------------------------------------------
::

    VENV=/mnt/local-ssd/bchuang/evox/.venv/bin/python
    export PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=4

    # numpy (CPU interpreter) -- all 9 configs, verified, merged JSON
    $VENV src/evogp/tree/dense_bench.py --backend numpy \
        --pops 64,200,1000 --rows 64,1024,10000 \
        --json src/evogp/tree/dense_bench_results.json

    # iree-cuda -- keep every process short (<= ~20 compiled executables each)
    $VENV src/evogp/tree/dense_bench.py --backend iree --smi-index 4 \
        --pops 64,200,1000 --rows 64,1024 --stage dense \
        --json src/evogp/tree/dense_bench_results.json
    $VENV src/evogp/tree/dense_bench.py --backend iree --smi-index 4 \
        --pops 64,200,1000 --rows 64,1024 --stage specialized \
        --json src/evogp/tree/dense_bench_results.json
    $VENV src/evogp/tree/dense_bench.py --backend iree --pops 64 --rows 64 \
        --stage real --json src/evogp/tree/dense_bench_results.json

    # cheap end-to-end smoke of the harness itself (numpy, 1 small case)
    $VENV src/evogp/tree/dense_bench.py --backend numpy --pops 64 --rows 64 \
        --repeats 1 --spec-sample 4

Notes / honest limitations
--------------------------
* Every timed call goes through ``Forest.SR_fitness`` (``per_row=False``);
  ``Forest.forward`` / ``batch_forward`` (``per_row=True``) are NOT covered.
* Device memory: the modeled per-chunk peak comes from ``dense_peak_bytes``;
  the measured numbers come from a background ``nvidia-smi
  --query-compute-apps`` sampler around ONE dedicated call per case
  (``--smi-index`` also samples the whole device) and from
  ``torch.cuda.max_memory_allocated``.  torch's allocator cannot see iree's
  allocations, so the two are reported side by side and never added up; the
  sampler is a lower bound (~50 ms period).
* Host memory: the numpy backend runs on the host and the dense graph keeps
  thousands of live ``(pop, rows)`` float32 arrays, so the dense path is also a
  host-memory bomb (~21.4 kB per pop x row element at ``L=32``, i.e. the
  documented ``dense_peak_bytes`` model is ~50x optimistic there -- e.g.
  ``P=64 x N=10000`` dies with the model predicting 273 MB).  Each measured
  case records the host RSS delta (``/proc/self/statm``) and the harness skips
  cases above ``--max-host-mb`` (default 6000) instead of being killed.
* Medians only; the box is shared with a parallel validation executor, so
  wall-clock numbers carry host contention (each case stores its repeat spread).
* Synthetic SR-style data: ``input_len=4``, uniform inputs in ``[-1, 1]``, a
  smooth target, trees from ``Forest.random_generate`` under a fixed seed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:  # allow `python src/evogp/tree/dense_bench.py` without PYTHONPATH
    sys.path.insert(0, _SRC)

from evogp.tree import Forest, GenerateDescriptor  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402

# ---------------------------------------------------------------------------
# comparison + formatting helpers (small and local: the spike scripts are owned
# by another workstream and are deliberately NOT imported here)
# ---------------------------------------------------------------------------
_RTOL = 1e-4
_ATOL = 1e-4
_RESULTS_VERSION = 1


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _compare(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Mask-exact, 1e-4-tolerant comparison of two float32 result arrays."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {a.shape} vs {b.shape}"}
    nan_ok = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    inf_a, inf_b = np.isinf(a), np.isinf(b)
    inf_ok = bool(np.array_equal(inf_a, inf_b)) and bool(
        np.array_equal(a[inf_a], b[inf_b])
    )
    finite = np.isfinite(a) & np.isfinite(b)
    max_abs = 0.0
    max_rel = 0.0
    tol_ok = True
    if finite.any():
        fa, fb = a[finite].astype(np.float64), b[finite].astype(np.float64)
        abs_diff = np.abs(fa - fb)
        max_abs = float(abs_diff.max())
        max_rel = float((abs_diff / np.maximum(np.abs(fb), 1e-45)).max())
        tol_ok = bool(np.all(abs_diff <= _ATOL + _RTOL * np.abs(fb)))
    return {
        "ok": bool(nan_ok and inf_ok and tol_ok),
        "nan_mask_ok": nan_ok,
        "inf_mask_ok": inf_ok,
        "finite_tol_ok": tol_ok,
        "n": int(a.size),
        "n_nan": int(np.isnan(a).sum()),
        "n_inf": int(np.isinf(a).sum()),
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def _fmt(value: Any, width: int = 9) -> str:
    if value is None:
        return " " * width
    if isinstance(value, str):
        return value[:width].rjust(width)
    if isinstance(value, float):
        if value != value:
            return "nan".rjust(width)
        if value >= 1e6:
            return f"{value:.3g}".rjust(width)
        if value >= 100:
            return f"{value:.1f}".rjust(width)
        return f"{value:.3f}".rjust(width)
    return str(value).rjust(width)


def _append_unique(items: List[str], value: str) -> List[str]:
    return items if value in items else items + [value]


def _git_head() -> Optional[str]:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _rss_mb() -> float:
    """Current host RSS of this process (MiB) -- the numpy backend is host-side."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * float(os.sysconf("SC_PAGE_SIZE")) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# nvidia-smi sampler (device memory the torch allocator cannot see)
# ---------------------------------------------------------------------------
class SmiSampler(threading.Thread):
    """Peak device memory of THIS process (and of GPU ``index``) around a call.

    Sampling ``nvidia-smi`` in a background thread sees the CUDA allocations of
    every runtime in the process (torch AND iree); the ~50 ms period can miss a
    short peak, so the value is a labelled lower bound.
    """

    def __init__(self, index: Optional[int], interval: float = 0.05):
        super().__init__(daemon=True)
        self.index = index
        self.interval = interval
        self.pid = os.getpid()
        self.peak_proc = 0
        self.peak_gpu = 0
        self.samples = 0
        self._stop_event = threading.Event()

    def _sample(self) -> None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == self.pid:
                    self.peak_proc = max(self.peak_proc, int(parts[1]))
        except Exception:
            pass
        if self.index is not None:
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.index),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                for value in out.split():
                    if value.strip().isdigit():
                        self.peak_gpu = max(self.peak_gpu, int(value))
            except Exception:
                pass
        self.samples += 1

    def run(self) -> None:  # pragma: no cover - thread body
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        self.join(timeout=10)
        return {
            "peak_proc_mb": round(self.peak_proc / 1024.0, 2),
            "peak_gpu_mb": (
                round(self.peak_gpu / 1024.0, 2) if self.index is not None else None
            ),
            "samples": self.samples,
        }


# ---------------------------------------------------------------------------
# public-API plumbing
# ---------------------------------------------------------------------------
def _apply(mode: str, backend: str) -> None:
    """Set the two documented env switches (both are read per call)."""
    os.environ["EVOGP_EVAL_MODE"] = mode
    os.environ["EVOGP_ETL_BACKEND"] = backend


def _sr_fitness(
    forest: Forest, inputs: torch.Tensor, labels: torch.Tensor
) -> np.ndarray:
    """One public-API fitness call (CUDA tensors in, host array out)."""
    out = forest.SR_fitness(inputs, labels)
    _sync()
    return out.detach().cpu().numpy()


def _time_one(fn: Callable[[], np.ndarray]) -> Tuple[float, np.ndarray]:
    _sync()
    start = time.perf_counter()
    result = fn()
    _sync()
    return time.perf_counter() - start, result


def _timed(
    fn: Callable[[], np.ndarray], repeats: int, warmup: int
) -> Tuple[List[float], np.ndarray]:
    """``1 + warmup`` untimed calls, then ``repeats`` timed calls (median later)."""
    result = fn()
    _sync()
    for _ in range(warmup):
        result = fn()
    _sync()
    times: List[float] = []
    for _ in range(repeats):
        seconds, result = _time_one(fn)
        times.append(seconds)
    return times, result


def _tree_length_stats(forest: Forest) -> Dict[str, float]:
    lengths = forest.batch_subtree_size[:, 0].detach().cpu().numpy().astype(np.int64)
    return {
        "min": int(lengths.min()),
        "median": float(np.median(lengths)),
        "mean": round(float(lengths.mean()), 2),
        "max": int(lengths.max()),
        "p90": float(np.percentile(lengths, 90)),
    }


def _program_stats(forest: Forest) -> Dict[str, Any]:
    """#distinct compiled programs (and constant trees) via the engine's decoder."""
    population = E.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    programs = [tree.program for tree in population.trees]
    nodes = [tree.node_count for tree in population.trees]
    return {
        "distinct_programs": int(len({p for p in programs if p is not None})),
        "constant_trees": int(sum(1 for p in programs if p is None)),
        "tree_nodes_median": float(np.median(nodes)),
        "tree_nodes_max": int(np.max(nodes)),
    }


def _make_case_data(
    rows: int, input_len: int, seed: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic SR-style data: uniform inputs, smooth finite target."""
    generator = np.random.default_rng(seed * 1000003 + rows)
    x = generator.uniform(-1.0, 1.0, size=(rows, input_len)).astype(np.float32)
    base = x[:, 0].copy()
    for column in range(1, input_len):
        base = base + (0.5 ** column) * x[:, column]
    y = (np.sin(base) + 0.25 * base * base + 0.1).astype(np.float32)
    inputs = torch.tensor(x, dtype=torch.float32, device=device)
    labels = torch.tensor(y[:, None], dtype=torch.float32, device=device)
    return inputs, labels


def _dense_plan(
    forest: Forest,
    rows: int,
    max_tree_len: int,
    budget_mb: float,
    update_mode: str,
    backend: str,
) -> Dict[str, Any]:
    """The documented plan/chunk/memory model of the dense path (public API)."""
    plan = D.plan_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        rows,
        per_row=False,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=backend,
    )
    out_mb = 4.0 * forest.pop_size * rows / (1024.0 * 1024.0)
    return {
        **D.describe_plan(plan),
        "peak_mb": round(plan.peak_bytes / (1024.0 * 1024.0), 3),
        "out_mb": round(out_mb, 3),
        "budget_mb": budget_mb,
    }


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------
class Bench:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda"
        self.backend = args.backend
        self.input_len = args.input_len
        self.max_tree_len = args.max_tree_len
        self.json_path = args.json
        self.results: Dict[str, Any] = self._load_results()
        self._pops: Dict[int, Forest] = {}
        self._data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # calibration state (per process)
        self._dense_points: Dict[int, List[Tuple[int, float]]] = {}
        self._dense_compiles: List[float] = []
        self._spec_samples: Dict[int, Dict[str, Any]] = {}
        # host-memory model of the dense numpy path: bytes per (pop x row) element,
        # calibrated from the measured host RSS deltas (default from the L=32 probe)
        self._host_measurements: List[float] = []

    # -- JSON (merged across the per-stage processes) -----------------------
    def _load_results(self) -> Dict[str, Any]:
        if self.json_path and os.path.exists(self.json_path):
            try:
                with open(self.json_path) as handle:
                    return json.load(handle)
            except Exception as exc:  # corrupt file -> start over, loudly
                print(f"WARNING: cannot read {self.json_path}: {exc}", flush=True)
        return {"version": _RESULTS_VERSION, "meta": {}, "cases": {}}

    def store(self) -> None:
        if not self.json_path:
            return
        self.results["version"] = _RESULTS_VERSION
        meta = self.results.setdefault("meta", {})
        meta["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["commands"] = _append_unique(meta.get("commands", []), " ".join(sys.argv))
        meta["env"] = {
            key: os.environ.get(key)
            for key in (
                "EVOGP_EVAL_MODE",
                "EVOGP_ETL_BACKEND",
                "EVOGP_DENSE_MAX_TREE_LEN",
                "EVOGP_DENSE_BUDGET_MB",
                "EVOGP_DENSE_UPDATE_MODE",
                "EVOGP_ETL_IREE_OPT_LEVEL",
                "ETL_DISABLE_LOCATIONS",
                "CUDA_VISIBLE_DEVICES",
            )
        }
        meta["git_head"] = _git_head()
        with open(self.json_path, "w") as handle:
            json.dump(self.results, handle, indent=2, sort_keys=True)

    def _case_record(self, key: str) -> Dict[str, Any]:
        return self.results.setdefault("cases", {}).setdefault(key, {})

    # -- populations / data -------------------------------------------------
    def population(self, pop_size: int) -> Forest:
        if pop_size not in self._pops:
            descriptor = GenerateDescriptor(
                max_tree_len=self.max_tree_len,
                input_len=self.input_len,
                output_len=1,
                using_funcs=["+", "-", "*", "/"],
                max_layer_cnt=self.args.max_layer_cnt,
                layer_leaf_prob=self.args.layer_leaf_prob,
                const_prob=0.5,
                const_range=(-5, 5),
                sample_cnt=8,
            )
            torch.manual_seed(self.args.seed)
            self._pops[pop_size] = Forest.random_generate(
                pop_size=pop_size, descriptor=descriptor
            )
            _sync()
        return self._pops[pop_size]

    def data(self, rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if rows not in self._data:
            self._data[rows] = _make_case_data(
                rows, self.input_len, self.args.seed, self.device
            )
        return self._data[rows]

    # -- cost model ---------------------------------------------------------
    def _dense_unit_ms(self) -> float:
        """ms per (pop x row) element, calibrated from the measured cases."""
        points = [
            (pop * rows, seconds)
            for pop, pairs in self._dense_points.items()
            for rows, seconds in pairs
        ]
        if points:
            return 1000.0 * sum(sec for _, sec in points) / sum(n for n, _ in points)
        return self.args.default_unit_ms

    def _dense_fit(self, pop_size: int) -> Tuple[float, float, str]:
        """``(a, b, note)`` of ``t_ms ~= a + b * rows`` for this pop size."""
        pairs = sorted(self._dense_points.get(pop_size, []))
        if len(pairs) >= 2:
            rows = np.asarray([p[0] for p in pairs], dtype=np.float64)
            times = np.asarray([1000.0 * p[1] for p in pairs], dtype=np.float64)
            if np.ptp(rows) > 0:
                slope, intercept = np.polyfit(rows, times, 1)
                return (
                    float(intercept),
                    float(slope),
                    f"fit(a={intercept:.3g}ms,b={slope:.4g}ms/row)",
                )
        if pairs:
            rows, seconds = pairs[-1]
            return 0.0, 1000.0 * seconds / rows, f"fit(origin,P={pop_size},N={rows})"
        return 0.0, self._dense_unit_ms(), "fit(unit-cost)"

    def _dense_estimate_ms(self, pop_size: int, rows: int) -> Tuple[float, str]:
        intercept, slope, note = self._dense_fit(pop_size)
        return intercept + slope * rows, note

    def _host_bytes_per_pn(self) -> float:
        """Host RSS bytes per (pop x row) element (min of the measured cases)."""
        if self._host_measurements:
            return min(self._host_measurements)
        return float(self.args.default_host_bytes_per_pop_row)

    def _compile_estimate_s(self, kind: str) -> float:
        if kind == "dense" and self._dense_compiles:
            return 1.5 * max(self._dense_compiles)
        return self.args.compile_defaults[kind]

    def _repeats_for(self, run_seconds: float) -> int:
        if run_seconds <= 2.0:
            return max(1, self.args.repeats)
        if run_seconds <= 20.0:
            return max(1, min(self.args.repeats, 2))
        return 1

    # -- dense stage --------------------------------------------------------
    def dense_case(
        self,
        forest: Forest,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        rows: int,
        record: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        pop_size = forest.pop_size
        est_run_s = self._dense_estimate_ms(pop_size, rows)[0] / 1000.0
        repeats = self._repeats_for(est_run_s)
        est_total_s = self._compile_estimate_s("dense") + (
            repeats + self.args.warmup + 2
        ) * est_run_s
        host_bytes_per_pn = self._host_bytes_per_pn()
        chunk_elements = max(1, record["plan"]["p_chunk"]) * max(
            1, record["plan"]["n_chunk"]
        )
        host_mb = host_bytes_per_pn * chunk_elements / (1024.0 * 1024.0)
        if host_mb > self.args.max_host_mb:
            estimate_ms, note = self._dense_estimate_ms(pop_size, rows)
            record["dense"] = {
                "status": "skipped_host_memory",
                "warm_ms": round(estimate_ms, 3),
                "warm_repeats": 0,
                "source": note,
                "host_estimate_mb": round(host_mb, 1),
                "host_model_bytes_per_chunk_element": round(host_bytes_per_pn, 1),
                "note": (
                    f"predicted per-chunk host RSS {host_mb:.0f} MB > "
                    f"--max-host-mb {self.args.max_host_mb:.0f} MB (the numpy "
                    f"interpreter keeps ~{host_bytes_per_pn:.0f} bytes per pop x row "
                    f"element, chunk {record['plan']['p_chunk']}x"
                    f"{record['plan']['n_chunk']} x {record['plan']['p_chunks']}*"
                    f"{record['plan']['n_chunks']}); warm_ms from {note}"
                ),
            }
            print(
                f"  dense   SKIP (host RSS model {host_mb:.0f} MB) -> extrapolated "
                f"{estimate_ms:.1f} ms ({note})",
                flush=True,
            )
            return None
        if est_total_s > self.args.max_config_seconds:
            estimate_ms, note = self._dense_estimate_ms(pop_size, rows)
            record["dense"] = {
                "status": "extrapolated",
                "warm_ms": round(estimate_ms, 3),
                "warm_repeats": 0,
                "source": note,
                "estimate_s": round(est_total_s, 1),
                "note": (
                    f"estimated {est_total_s:.0f}s > --max-config-seconds "
                    f"{self.args.max_config_seconds:.0f}s; warm_ms from {note}"
                ),
            }
            print(
                f"  dense   SKIP -> extrapolated {estimate_ms:.1f} ms "
                f"({note}; est {est_total_s:.0f}s)",
                flush=True,
            )
            return None

        _apply("dense", self.backend)
        E.clear_cache()
        gc.collect()
        stats_before = E.cache_stats()
        rss_before = _rss_mb()
        cold_s, cold_out = _time_one(lambda: _sr_fitness(forest, inputs, labels))
        rss_cold = _rss_mb()
        stats_after = E.cache_stats()
        misses = stats_after["misses"] - stats_before["misses"]

        memory: Dict[str, Any] = {
            "smi": None,
            "torch_peak_alloc_mb": None,
            "model_peak_mb": record["plan"]["peak_mb"],
            "model_out_mb": record["plan"]["out_mb"],
            "host_rss_before_mb": round(rss_before, 1),
            "host_rss_after_cold_mb": round(rss_cold, 1),
        }
        if self.args.smi:
            torch.cuda.reset_peak_memory_stats()
            sampler = SmiSampler(self.args.smi_index)
            sampler.start()
            try:
                _, memory_out = _time_one(lambda: _sr_fitness(forest, inputs, labels))
            finally:
                memory["smi"] = sampler.stop()
            memory["torch_peak_alloc_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2
            )
            memory["memory_call_matches"] = _compare(cold_out, memory_out)

        times, out = _timed(
            lambda: _sr_fitness(forest, inputs, labels), repeats, self.args.warmup
        )
        warm_s = _median(times)
        rss_after = _rss_mb()
        delta_bytes = max(0.0, rss_after - rss_before) * 1024.0 * 1024.0
        if pop_size * rows >= 65536 and delta_bytes > 0:
            # only run-dominated cases calibrate the host model: small cases are
            # dominated by the (rows-independent) compile allocation
            self._host_measurements.append(delta_bytes / (pop_size * rows))
        memory["host_rss_after_warm_mb"] = round(rss_after, 1)
        memory["host_rss_delta_mb"] = round(delta_bytes / (1024.0 * 1024.0), 1)
        memory["host_bytes_per_pop_row"] = round(
            delta_bytes / max(1, pop_size * rows), 1
        )
        memory["host_note"] = (
            "current host RSS (/proc/self/statm) before the cold call / after the cold "
            "call / after the timed calls; the delta includes the compile and the run "
            "and is the calibration of the harness' own host-memory guard"
        )
        self._dense_points.setdefault(pop_size, []).append((rows, warm_s))
        self._dense_compiles.append(max(0.0, cold_s - warm_s))
        if (record.get("dense") or {}).get("status") == "measured":
            record.setdefault("dense_previous", []).append(record["dense"])
        record["dense"] = {
            "status": "measured",
            "compile_ms": round(1000.0 * max(0.0, cold_s - warm_s), 3),
            "cold_ms": round(1000.0 * cold_s, 3),
            "warm_ms": round(1000.0 * warm_s, 3),
            "warm_ms_all": [round(1000.0 * t, 3) for t in times],
            "warm_repeats": repeats,
            "warmup": self.args.warmup,
            "cache_misses": int(misses),
            "cache_hits_after_cold": int(stats_after["hits"] - stats_before["hits"]),
            "finite_fraction": float(np.isfinite(out).mean()),
            "memory": memory,
            "source": "measured",
        }
        if misses != 1:
            print(f"  WARNING: dense cold call built {misses} executables (expected 1)")
        print(
            f"  dense   cold {1000.0 * cold_s:8.1f} ms "
            f"(compile {1000.0 * max(0.0, cold_s - warm_s):8.1f}) "
            f"warm {1000.0 * warm_s:9.2f} ms  builds={misses} "
            f"host_rss +{memory['host_rss_delta_mb']:.0f} MB "
            f"({memory['host_bytes_per_pop_row']:.0f} B/pop-row)",
            flush=True,
        )
        return out

    # -- specialized stage --------------------------------------------------
    def specialized_case(
        self,
        forest: Forest,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        rows: int,
        record: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        _apply("specialized", self.backend)
        decode_ms = self._decode_ms(forest)
        if self.backend == "numpy":
            return self._specialized_full(forest, inputs, labels, decode_ms, record)

        # iree: per-tree sample model (full-population cold is infeasible there)
        distinct = record["programs"]["distinct_programs"]
        sample = self._sample_model(rows, decode_ms)
        model_warm_ms = decode_ms + forest.pop_size * sample["per_tree_warm_ms_median"]
        model_compile_ms = distinct * sample["per_tree_compile_ms_median"]
        record["specialized"] = {
            "status": "extrapolated",
            "source": "per-tree sample x pop_size",
            "warm_ms": round(model_warm_ms, 3),
            "compile_ms": round(model_compile_ms, 3),
            "model_warm_ms": round(model_warm_ms, 3),
            "model_compile_ms": round(model_compile_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "distinct_programs": distinct,
            "sample": sample,
            "note": (
                "warm_ms = decode_ms + pop_size * median(per-tree warm run); "
                "compile_ms = #distinct programs * median(per-tree compile ms); "
                "per-tree numbers measured once per N on the reference population"
            ),
        }
        print(
            f"  spec    model warm {model_warm_ms:9.2f} ms  compile {model_compile_ms:8.1f} ms "
            f"(per-tree {sample['per_tree_warm_ms_median']:.2f} warm / "
            f"{sample['per_tree_compile_ms_median']:.1f} compile ms x {distinct} distinct)",
            flush=True,
        )
        return None

    def _decode_ms(self, forest: Forest) -> float:
        times: List[float] = []
        for _ in range(max(1, min(3, self.args.repeats))):
            _sync()
            start = time.perf_counter()
            E.decode_population(
                forest.batch_node_value,
                forest.batch_node_type,
                forest.batch_subtree_size,
                forest.input_len,
                forest.output_len,
            )
            times.append(time.perf_counter() - start)
        return 1000.0 * _median(times)

    def _specialized_full(
        self,
        forest: Forest,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        decode_ms: float,
        record: Dict[str, Any],
    ) -> np.ndarray:
        _apply("specialized", "numpy")
        E.clear_cache()
        gc.collect()
        stats_before = E.cache_stats()
        cold_s, _ = _time_one(lambda: _sr_fitness(forest, inputs, labels))
        stats_after = E.cache_stats()
        repeats = self._repeats_for(cold_s)
        times, out = _timed(
            lambda: _sr_fitness(forest, inputs, labels), repeats, self.args.warmup
        )
        warm_s = _median(times)
        record["specialized"] = {
            "status": "measured",
            "cold_ms": round(1000.0 * cold_s, 3),
            "warm_ms": round(1000.0 * warm_s, 3),
            "warm_ms_all": [round(1000.0 * t, 3) for t in times],
            "warm_repeats": repeats,
            "compile_ms": round(1000.0 * max(0.0, cold_s - warm_s), 3),
            "cache_misses": int(stats_after["misses"] - stats_before["misses"]),
            "cache_hits": int(stats_after["hits"] - stats_before["hits"]),
            "decode_ms": round(decode_ms, 3),
            "finite_fraction": float(np.isfinite(out).mean()),
            "source": "measured",
        }
        print(
            f"  spec    cold {1000.0 * cold_s:8.1f} ms "
            f"(compile {1000.0 * max(0.0, cold_s - warm_s):8.1f}) "
            f"warm {1000.0 * warm_s:9.2f} ms  builds={record['specialized']['cache_misses']}",
            flush=True,
        )
        if self.args.numpy_sample:
            sample = self._sample_model(forest.pop_size, decode_ms, forest=forest)
            model_warm = decode_ms + forest.pop_size * sample["per_tree_warm_ms_median"]
            model_compile = (
                record["programs"]["distinct_programs"]
                * sample["per_tree_compile_ms_median"]
            )
            sample["model_warm_ms"] = round(model_warm, 3)
            sample["model_compile_ms"] = round(model_compile, 3)
            sample["actual_warm_ms"] = record["specialized"]["warm_ms"]
            sample["actual_compile_ms"] = record["specialized"]["compile_ms"]
            record["specialized"]["sample_model"] = sample
            print(
                f"  spec    model (same method as iree) warm {model_warm:9.2f} ms vs actual "
                f"{record['specialized']['warm_ms']:.2f} | compile {model_compile:8.1f} vs actual "
                f"{record['specialized']['compile_ms']:.1f}",
                flush=True,
            )
        return out

    def _sample_model(
        self, rows: int, decode_ms: float, forest: Optional[Forest] = None
    ) -> Dict[str, Any]:
        """Per-tree compile/run sample -> full-population model (one per row count)."""
        if rows in self._spec_samples:
            return self._spec_samples[rows]
        args = self.args
        sample_pop = int(args.spec_sample_pop or args.pops[0])
        reference = forest if forest is not None else self.population(sample_pop)
        inputs, labels = self.data(rows)
        sample_size = min(args.spec_sample, reference.pop_size)
        indices = sorted(
            {int(round(i)) for i in np.linspace(0, reference.pop_size - 1, sample_size)}
        )
        _apply("specialized", self.backend)
        per_tree_compile: List[float] = []
        per_tree_warm: List[float] = []
        per_tree_misses: List[int] = []
        for index in indices:
            single = reference[index].to_forest()
            E.clear_cache()
            gc.collect()
            stats_before = E.cache_stats()
            cold_s, _ = _time_one(lambda single=single: _sr_fitness(single, inputs, labels))
            stats_after = E.cache_stats()
            times, _ = _timed(
                lambda single=single: _sr_fitness(single, inputs, labels),
                1 if args.sample_repeats < 1 else args.sample_repeats,
                1,
            )
            warm_s = _median(times)
            per_tree_compile.append(max(0.0, cold_s - warm_s))
            per_tree_warm.append(warm_s)
            per_tree_misses.append(stats_after["misses"] - stats_before["misses"])
        entry = {
            "sample_pop": reference.pop_size,
            "sample_size": len(indices),
            "sample_indices": indices,
            "cache_misses_per_tree": per_tree_misses,
            "per_tree_warm_ms_median": round(1000.0 * _median(per_tree_warm), 3),
            "per_tree_warm_ms_all": [round(1000.0 * t, 3) for t in per_tree_warm],
            "per_tree_compile_ms_median": round(1000.0 * _median(per_tree_compile), 3),
            "per_tree_compile_ms_all": [round(1000.0 * t, 3) for t in per_tree_compile],
            "decode_ms": round(decode_ms, 3),
            "method": (
                "single-tree Forest (Forest[i].to_forest()), cache cleared per tree: "
                "cold one call, warm median of --sample-repeats"
            ),
        }
        if self.backend != "numpy" and any(m != 1 for m in per_tree_misses):
            print(f"  WARNING: per-tree cold calls built {per_tree_misses} executables")
        self._spec_samples[rows] = entry
        print(
            f"  spec    sample: per-tree {entry['per_tree_warm_ms_median']:.3f} ms warm / "
            f"{entry['per_tree_compile_ms_median']:.1f} ms compile "
            f"({len(indices)} trees, pop {reference.pop_size}, N={rows})",
            flush=True,
        )
        return entry

    def real_small_pop(self, rows: int) -> None:
        """REAL end-to-end specialized call at a small population (model check)."""
        reference = self.population(int(self.args.spec_sample_pop or self.args.pops[0]))
        pop_size = min(self.args.spec_real_pop, reference.pop_size)
        forest = reference[:pop_size]
        inputs, labels = self.data(rows)
        _apply("specialized", self.backend)
        E.clear_cache()
        gc.collect()
        stats_before = E.cache_stats()
        cold_s, out = _time_one(lambda: _sr_fitness(forest, inputs, labels))
        stats_after = E.cache_stats()
        times, warm_out = _timed(
            lambda: _sr_fitness(forest, inputs, labels), max(1, self.args.repeats), 1
        )
        warm_s = _median(times)
        key = f"{self.backend}_P{pop_size}_N{rows}"
        record = self._case_record(key)
        record.update(
            {"backend": self.backend, "pop": pop_size, "rows": rows, "seed": self.args.seed}
        )
        model = self._model_for(rows)
        real_per_tree_ms = 1000.0 * warm_s / pop_size
        real = {
            "pool_pop": reference.pop_size,
            "pop": pop_size,
            "cold_ms": round(1000.0 * cold_s, 3),
            "warm_ms": round(1000.0 * warm_s, 3),
            "compile_ms": round(1000.0 * max(0.0, cold_s - warm_s), 3),
            "cache_misses": int(stats_after["misses"] - stats_before["misses"]),
            "warm_repeats": max(1, self.args.repeats),
            "finite_fraction": float(np.isfinite(warm_out).mean()),
            "per_tree_warm_ms_effective": round(real_per_tree_ms, 3),
        }
        if model is not None:
            real["model_per_tree_warm_ms"] = model["per_tree_warm_ms_median"]
            real["model_per_tree_warm_ratio"] = round(
                model["per_tree_warm_ms_median"] / real_per_tree_ms, 3
            ) if real_per_tree_ms > 0 else None
            real["model_source_case"] = model["case"]
        record["specialized"] = {
            "status": "measured",
            "source": "measured(small pop)",
            "cold_ms": real["cold_ms"],
            "warm_ms": real["warm_ms"],
            "compile_ms": real["compile_ms"],
            "cache_misses": real["cache_misses"],
            "warm_repeats": real["warm_repeats"],
            "finite_fraction": real["finite_fraction"],
            "real_small_pop": real,
        }
        record["verify"] = self.verify(forest, inputs, labels, None, out)
        self.store()
        print(
            f"[{key}] REAL small-pop specialized: cold {real['cold_ms']:.1f} ms "
            f"(compile {real['compile_ms']:.1f}) warm {real['warm_ms']:.3f} ms "
            f"builds={real['cache_misses']} -> effective "
            f"{real['per_tree_warm_ms_effective']:.3f} ms/tree"
            + (
                f" vs model {real['model_per_tree_warm_ms']:.3f} ms/tree "
                f"(x{real['model_per_tree_warm_ratio']})"
                if model is not None
                else ""
            ),
            flush=True,
        )

    def _model_for(self, rows: int) -> Optional[Dict[str, Any]]:
        """The per-tree sample model stored by the specialized stage (any pop)."""
        for case in self.results.get("cases", {}).values():
            if case.get("backend") != self.backend or case.get("rows") != rows:
                continue
            specialized = case.get("specialized") or {}
            sample = specialized.get("sample") or specialized.get("sample_model")
            if sample and "per_tree_warm_ms_median" in sample:
                return {**sample, "case": f"P{case.get('pop')}_N{case.get('rows')}"}
        return None

    # -- one config ---------------------------------------------------------
    def run_case(self, pop_size: int, rows: int) -> Dict[str, Any]:
        key = f"{self.backend}_P{pop_size}_N{rows}"
        record = self._case_record(key)
        record.update(
            {
                "backend": self.backend,
                "pop": pop_size,
                "rows": rows,
                "seed": self.args.seed,
                "input_len": self.input_len,
                "max_tree_len": self.max_tree_len,
                "max_layer_cnt": self.args.max_layer_cnt,
                "budget_mb": self.args.budget_mb,
                "update_mode": self.args.update_mode,
            }
        )
        forest = self.population(pop_size)
        inputs, labels = self.data(rows)
        record["tree_lengths"] = _tree_length_stats(forest)
        record["programs"] = _program_stats(forest)
        previous_plan = record.get("plan")
        try:
            record["plan"] = _dense_plan(
                forest,
                rows,
                self.max_tree_len,
                self.args.budget_mb,
                self.args.update_mode,
                self.backend,
            )
            dense_ok = True
            if previous_plan and previous_plan.get("budget_mb") != self.args.budget_mb:
                record.setdefault("plans_previous", []).append(previous_plan)
        except D.DenseUnsupported as exc:
            record["plan"] = {"dense_supported": False, "reason": str(exc)}
            dense_ok = False
        record["plan"]["dense_supported"] = dense_ok
        print(
            f"[{key}] L={record['plan'].get('trace_len')} "
            f"tree_len max {record['tree_lengths']['max']} "
            f"distinct {record['programs']['distinct_programs']}/{pop_size} "
            f"chunk {record['plan'].get('p_chunk')}x{record['plan'].get('n_chunk')}"
            f" x {record['plan'].get('p_chunks')}*{record['plan'].get('n_chunks')} "
            f"peak {record['plan'].get('peak_mb')} MB",
            flush=True,
        )

        dense_out = None
        if self.args.stage in ("all", "dense"):
            if dense_ok:
                dense_out = self.dense_case(forest, inputs, labels, rows, record)
            else:
                record["dense"] = {
                    "status": "unsupported",
                    "reason": record["plan"]["reason"],
                }
                print("  dense   NOT dense-supported (would fall back)", flush=True)

        spec_out = None
        if self.args.stage in ("all", "specialized"):
            spec_out = self.specialized_case(forest, inputs, labels, rows, record)

        if not self.args.skip_verify:
            record["verify"] = self.verify(forest, inputs, labels, dense_out, spec_out)

        dense_ms = (record.get("dense") or {}).get("warm_ms")
        spec_ms = (record.get("specialized") or {}).get("warm_ms")
        if dense_ms and spec_ms:
            record["ratio_warm_spec_over_dense"] = round(spec_ms / dense_ms, 3)
        self.store()
        return record

    def verify(
        self,
        forest: Forest,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        dense_out: Optional[np.ndarray],
        spec_out: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        pop_size = forest.pop_size
        rows = int(inputs.shape[0])
        if pop_size <= 64 and rows <= 1024:
            indices = list(range(pop_size))
            full = True
        else:
            indices = sorted(
                {
                    int(round(i))
                    for i in np.linspace(
                        0, pop_size - 1, min(self.args.verify_rows, pop_size)
                    )
                }
            )
            full = False
        index_array = np.asarray(indices, dtype=np.int64)
        try:
            _apply("specialized", "numpy")  # independent: numpy interpreter reference
            reference = _sr_fitness(forest[index_array], inputs, labels)
        finally:
            _apply("specialized", self.backend)
        result: Dict[str, Any] = {
            "rows_checked": len(indices),
            "full_population": full,
            "reference": "numpy-specialized (public API, same tensors)",
            "reference_finite_fraction": float(np.isfinite(reference).mean()),
        }
        if dense_out is not None:
            result["dense"] = _compare(dense_out[index_array], reference)
        if spec_out is not None:
            result["specialized"] = _compare(spec_out[index_array], reference)
        if dense_out is not None and spec_out is not None:
            result["dense_vs_specialized"] = _compare(dense_out, spec_out)
        result["verified"] = bool(
            result.get("dense", {"ok": True})["ok"]
            and result.get("specialized", {"ok": True})["ok"]
        )
        flag = "OK      " if result["verified"] else "MISMATCH"
        max_rel = result.get("dense", {}).get("max_rel", 0.0)
        print(
            f"  verify  {flag} {len(indices)}/{pop_size} trees (full={full}) "
            f"dense max_rel {max_rel:.3g} ref_finite "
            f"{result['reference_finite_fraction']:.3f}",
            flush=True,
        )
        return result


# ---------------------------------------------------------------------------
# CLI + the report table
# ---------------------------------------------------------------------------
def _parse_int_list(text: str) -> List[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="numpy", choices=["numpy", "iree"])
    parser.add_argument("--pops", default="64,200,1000", help="comma-separated pop sizes")
    parser.add_argument(
        "--rows", default="64,1024,10000", help="comma-separated data row counts"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "dense", "specialized", "real"],
        help="which evaluator to measure (use separate processes on iree)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="max timed repeats (median)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sample-repeats", type=int, default=3, help="per-tree warm repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-len", type=int, default=4)
    parser.add_argument("--max-tree-len", type=int, default=32)
    parser.add_argument("--max-layer-cnt", type=int, default=5)
    parser.add_argument("--layer-leaf-prob", type=float, default=0.2)
    parser.add_argument("--budget-mb", type=float, default=2048.0)
    parser.add_argument("--update-mode", default="grow", choices=list(D.UPDATE_MODES))
    parser.add_argument(
        "--max-config-seconds",
        type=float,
        default=150.0,
        help="skip (and extrapolate) a case whose estimated cost exceeds this",
    )
    parser.add_argument(
        "--max-host-mb",
        type=float,
        default=6000.0,
        help="skip a dense case whose predicted host RSS exceeds this (numpy backend)",
    )
    parser.add_argument(
        "--default-host-bytes-per-pop-row",
        type=float,
        default=21400.0,
        help=(
            "host-RSS model: bytes per (pop x row) element before calibration; "
            "the dense numpy graph keeps ~5.4k live (pop, row) float32 arrays per "
            "traced position chain, measured 21.4 kB/element at L=32"
        ),
    )
    parser.add_argument("--spec-sample", type=int, default=12, help="trees sampled per case")
    parser.add_argument(
        "--spec-sample-pop",
        type=int,
        default=None,
        help="population the per-tree sample is drawn from (default: first --pops)",
    )
    parser.add_argument(
        "--spec-real-pop", type=int, default=16, help="pop of the real model-check call"
    )
    parser.add_argument("--verify-rows", type=int, default=32, help="trees in the reference subset")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--numpy-sample",
        action="store_true",
        help="also run the iree per-tree model on numpy (cross-check of the method)",
    )
    parser.add_argument("--smi", action="store_true", help="nvidia-smi memory sampling")
    parser.add_argument("--no-smi", action="store_true", help="disable nvidia-smi sampling")
    parser.add_argument(
        "--smi-index", type=int, default=None, help="physical GPU index for nvidia-smi"
    )
    parser.add_argument(
        "--default-unit-ms",
        type=float,
        default=None,
        help="fallback dense ms per (pop x row) element before calibration",
    )
    parser.add_argument("--json", default=None, help="merged results JSON (optional)")
    parser.add_argument("--list-cases", action="store_true", help="print the planned grid and exit")
    return parser


def _compile_defaults(backend: str) -> Dict[str, float]:
    if backend == "numpy":
        return {"dense": 0.05, "specialized": 0.02}
    return {"dense": 40.0, "specialized": 1.2}


def print_table(results: Dict[str, Any]) -> None:
    print()
    header = (
        f"{'backend':>8} {'pop':>6} {'rows':>6} | {'dense':>10} {'src':>14} "
        f"{'dcomp':>9} {'rssMB':>7} | {'spec':>10} {'src':>9} {'scomp':>10} | "
        f"{'ratio':>7} {'chunks':>7} {'ok':>4}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(results.get("cases", {}), key=_case_sort_key):
        case = results["cases"][key]
        dense = case.get("dense", {}) or {}
        spec = case.get("specialized", {}) or {}
        chunks = "-"
        if "p_chunk" in case.get("plan", {}):
            chunks = f"{case['plan']['p_chunks']}x{case['plan']['n_chunks']}"
        source = dense.get("source", "-")
        if dense.get("status") != "measured":
            source = str(dense.get("status", "-"))
        rss = (dense.get("memory") or {}).get("host_rss_delta_mb")
        if rss is None:
            rss = dense.get("host_estimate_mb")
        print(
            f"{case.get('backend', '?'):>8} {case.get('pop', '?'):>6} {case.get('rows', '?'):>6} | "
            f"{_fmt(dense.get('warm_ms'))} {str(source)[:14]:>14} "
            f"{_fmt(dense.get('compile_ms'))} {_fmt(rss, 7)} | {_fmt(spec.get('warm_ms'))} "
            f"{str(spec.get('source', spec.get('status', '-')))[:9]:>9} "
            f"{_fmt(spec.get('compile_ms'))} | "
            f"{_fmt(case.get('ratio_warm_spec_over_dense'), 7)} {chunks:>7} "
            f"{('yes' if case.get('verify', {}).get('verified') else 'NO'):>4}"
        )
    print()


def _case_sort_key(key: str) -> Tuple[str, int, int]:
    parts = key.split("_")
    try:
        return (parts[0], int(parts[1][1:]), int(parts[2][1:]))
    except Exception:
        return (key, 0, 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.pops = _parse_int_list(args.pops)
    args.rows = _parse_int_list(args.rows)
    if args.no_smi:
        args.smi = False
    elif args.backend == "iree":
        args.smi = True
    if args.default_unit_ms is None:
        args.default_unit_ms = 0.05 if args.backend == "numpy" else 0.04
    args.compile_defaults = _compile_defaults(args.backend)
    if args.list_cases:
        for pop in args.pops:
            for rows in args.rows:
                print(f"{args.backend}_P{pop}_N{rows}")
        return 0
    if not torch.cuda.is_available():
        print("ERROR: this harness needs a CUDA-visible device (the API is CUDA-bound)")
        return 2
    os.environ.setdefault("EVOGP_ETL_IREE_DEVICE", "0")
    os.environ.setdefault("EVOGP_DENSE_MAX_TREE_LEN", str(args.max_tree_len))
    os.environ["EVOGP_DENSE_BUDGET_MB"] = str(args.budget_mb)
    os.environ["EVOGP_DENSE_UPDATE_MODE"] = args.update_mode

    bench = Bench(args)
    print(
        f"dense_bench: backend={args.backend} stage={args.stage} seed={args.seed} "
        f"pops={args.pops} rows={args.rows} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    if args.stage == "real":
        for rows in args.rows:
            bench.real_small_pop(rows)
    else:
        for pop_size in args.pops:
            for rows in args.rows:
                bench.run_case(pop_size, rows)
                bench.store()

    print_table(bench.results)
    if args.json:
        print(f"results JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
