"""Shared plumbing for the ``test/benchmark/evoxc`` campaign.

This package times the CURRENT (evoxc-based) EvoGP engine against **the old
plain-PyTorch way of evaluating the same forests** -- no evoxc, no jit -- on
identical populations, in the same session, with a warm/steady-state protocol.

What lives here and nowhere else:

* the cell grid (population shape x tree length x input length x data rows) and
  the deterministic population builder every arm must share;
* the arm registry (torch arms + the evoxc engine routes) and the environment
  knobs each arm sets;
* the timing helper (sync-bracketed, warmup + reps, median/min/max/samples);
* the value digest used to prove cross-process numeric agreement;
* the environment block written into every record.

Nothing here imports ``evogp`` at module level: the campaign re-points
``PYTHONPATH`` at a tree root (``--tree-root``), so the import has to happen
inside the worker process after that root is in place.  ``import_evogp()`` is the
one accessor.

Run the campaign with ``python test/benchmark/evoxc/campaign.py --help``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "evogp-bench-evoxc-v1"

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
#: The evoxc engine route of each arm, as ``(EVOGP_EVAL_MODE, EVOGP_EVOXC_BACKEND)``.
#: ``None`` means "leave the environment alone" (the library default).
EVOXC_ARMS: Dict[str, Dict[str, Optional[str]]] = {
    # the shipped default: EVOGP_EVAL_MODE=auto, EVOGP_EVOXC_BACKEND=numpy
    "evoxc-auto": {"eval_mode": None, "backend": None, "family": "evoxc"},
    # auto routing with the GPU backend opted in (the interesting auto case)
    "evoxc-auto-iree": {"eval_mode": None, "backend": "iree", "family": "evoxc"},
    # one compiled evoxc graph per tree structure
    "evoxc-specialized-numpy": {
        "eval_mode": "specialized",
        "backend": "numpy",
        "family": "evoxc",
    },
    "evoxc-specialized-iree": {
        "eval_mode": "specialized",
        "backend": "iree",
        "family": "evoxc",
    },
    # one dense all-ops graph per population shape
    "evoxc-dense-iree": {"eval_mode": "dense", "backend": "iree", "family": "evoxc"},
}
TORCH_ARMS: Dict[str, Dict[str, Any]] = {
    "torch-per-tree": {"impl": "per_tree", "family": "torch"},
    "torch-batched": {"impl": "batched", "family": "torch"},
}
ALL_ARMS: List[str] = list(TORCH_ARMS) + list(EVOXC_ARMS)

#: Workloads. ``sr_fitness`` is the training hot path, ``batch_forward`` is the
#: public ``Forest.batch_forward`` value path (``(pop, N, out)``), ``forward`` is
#: the per-row value path (``(pop, out)``: tree ``i`` sees row ``i``), ``gp_step``
#: is one full GP generation (evaluate + select + crossover + mutate).
WORKLOADS = ("sr_fitness", "batch_forward", "forward", "gp_step")


# ---------------------------------------------------------------------------
# cell grid
# ---------------------------------------------------------------------------
POPS = (16, 64, 256, 1000)
TREE_LENS = (32, 64)
INPUT_LENS = (8, 32)
MAIN_ROWS = 1024
#: the second data-row value ("one N=64 column"): L=64/input_len=32 at every pop
N64_ROWS = 64


def cell(pop: int, tree_len: int, input_len: int, rows: int) -> Dict[str, Any]:
    return {
        "id": f"p{pop}-L{tree_len}-i{input_len}-N{rows}",
        "pop": pop,
        "tree_len": tree_len,
        "input_len": input_len,
        "rows": rows,
    }


def main_cells() -> List[Dict[str, Any]]:
    """The 4 x 2 x 2 grid at ``N = 1024`` (16 cells)."""
    return [
        cell(pop, tree_len, input_len, MAIN_ROWS)
        for pop in POPS
        for tree_len in TREE_LENS
        for input_len in INPUT_LENS
    ]


def n64_cells() -> List[Dict[str, Any]]:
    """The ``N = 64`` column: every pop at tree_len 64 / input_len 32."""
    return [cell(pop, 64, 32, N64_ROWS) for pop in POPS]


def parse_cell(text: str) -> Dict[str, Any]:
    """``p256-L64-i32-N1024`` -> the cell dict."""
    parts = dict(part.split("=", 1) if "=" in part else (part[0], part[1:]) for part in text.split("-"))
    try:
        return cell(int(parts["p"]), int(parts["L"]), int(parts["i"]), int(parts["N"]))
    except Exception as exc:  # noqa: BLE001 - a bad id is a usage error
        raise SystemExit(f"bad cell id {text!r}: {exc}")


# ---------------------------------------------------------------------------
# environment / imports
# ---------------------------------------------------------------------------
def import_evogp():
    """Import the library from whatever ``PYTHONPATH`` the caller set up."""
    import evogp.tree as tree  # noqa: F401  (the package under test)
    import evogp.algorithm as algorithm  # noqa: F401
    import evogp.pipeline as pipeline  # noqa: F401
    import evogp.problem as problem  # noqa: F401

    return tree, algorithm, pipeline, problem


def _git_head(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort provenance
        pass
    return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def env_knobs() -> Dict[str, str]:
    """Every ``EVOGP_*`` / ``EVOXC_*`` variable the engine can read, as set."""
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("EVOGP_") or key.startswith("EVOXC_")
    }


def env_block(tree_root: str, arm: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Environment provenance written at the head of every record."""
    import numpy as np
    import torch

    _, _, _, _ = (None, None, None, None)
    block: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree_root": os.path.abspath(tree_root),
        "tree_root_sha": _git_head(os.path.abspath(tree_root)),
        "cwd": os.getcwd(),
        "arm": arm,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "iree_base_compiler": _pkg_version("iree-base-compiler"),
        "iree_base_runtime": _pkg_version("iree-base-runtime"),
        "evoxc_path": None,
        "evoxc_sha": _git_head("/root/evoxc"),
        "evogp_file": None,
        "gpu": None,
        "gpu_total_memory_gb": None,
        "knobs": env_knobs(),
    }
    try:
        import evoxc

        block["evoxc_path"] = getattr(evoxc, "__file__", None)
        block["evoxc_version"] = getattr(evoxc, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        block["evoxc_import_error"] = repr(exc)
    try:
        import evogp

        block["evogp_file"] = getattr(evogp, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        block["evogp_import_error"] = repr(exc)
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            block["gpu"] = props.name
            block["gpu_total_memory_gb"] = round(props.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        block["gpu_error"] = repr(exc)
    if extra:
        block.update(extra)
    return block


# ---------------------------------------------------------------------------
# workload construction (ONE definition, shared by every arm)
# ---------------------------------------------------------------------------
#: The function set every population is generated from.  Restricted to the four
#: correctly-rounded arithmetic operators on purpose: with them the plain
#: float32 torch arms and the reference oracle agree bit-exactly (see
#: ``test/benchmark/evoxc/README.md`` "Numeric validation"), so a mismatch is a
#: real bug and not one-ulp transcendental noise.
def src_root() -> str:
    """The ``src`` directory of the *imported* evogp checkout (the ``--tree-root`` switch)."""
    import evogp

    return os.path.dirname(os.path.dirname(os.path.abspath(evogp.__file__)))


def checkout_root() -> str:
    """The git checkout that contains :func:`src_root` (``/tmp/evogp_preopt`` for a snapshot)."""
    return os.path.dirname(src_root())


DESCRIPTOR_FUNCS = ("+", "-", "*", "/")


def make_descriptor(tree_len: int, input_len: int, output_len: int = 1, using_funcs=None, max_layer_cnt: int = 5):
    """The canonical generation descriptor (mirrors ``src/evogp/sr_test.py``)."""
    from evogp.tree import GenerateDescriptor

    return GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=input_len,
        output_len=output_len,
        const_prob=0.5,
        out_prob=0.5,
        using_funcs=list(using_funcs or DESCRIPTOR_FUNCS),
        layer_leaf_prob=0.2,
        const_range=(-5, 5),
        sample_cnt=8,
        max_layer_cnt=max_layer_cnt,
    )


def build_workload(cell_spec: Dict[str, Any], seed: int = 0, using_funcs=None, max_layer_cnt: int = 5) -> Dict[str, Any]:
    """Deterministically build the population and data of one cell.

    ``torch.manual_seed(seed)`` + ``Forest.random_generate`` is the documented
    reproducibility recipe, so every arm in every process sees the same trees.
    """
    import torch
    from evogp.tree import Forest

    torch.manual_seed(seed)
    descriptor = make_descriptor(
        cell_spec["tree_len"], cell_spec["input_len"], using_funcs=using_funcs, max_layer_cnt=max_layer_cnt
    )
    forest = Forest.random_generate(cell_spec["pop"], descriptor)
    rows = cell_spec["rows"]
    x = torch.rand(rows, cell_spec["input_len"], device="cuda")
    labels = torch.rand(rows, 1, device="cuda")
    # per_row value path: one input row per tree
    x_per_row = torch.rand(cell_spec["pop"], cell_spec["input_len"], device="cuda")
    return {
        "forest": forest,
        "descriptor": descriptor,
        "x": x,
        "labels": labels,
        "x_per_row": x_per_row,
        "cell": cell_spec,
        "seed": seed,
        "using_funcs": list(using_funcs or DESCRIPTOR_FUNCS),
    }


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_call(fn, warmup: int = 2, reps: int = 5, stats=None) -> Dict[str, Any]:
    """Warm/steady-state timing of ``fn`` (sync-bracketed, per-call samples).

    The first call is timed separately and never counted in the warm statistics:
    that is where a cold compile lands.  ``stats`` (a zero-arg callable, e.g.
    ``evoxc_engine.cache_stats``) is sampled right after it, so the compile cost
    and the compile *misses* are both attributable to that one call.
    """
    import torch

    # one untimed call: this is where a first-call compile lands
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3
    after_first = stats() if stats is not None else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    ordered = sorted(samples)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "first_call_ms": first_ms,
        "cache_after_first_call": after_first,
        "warmup": warmup,
        "reps": reps,
        "samples_ms": samples,
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": sum(samples) / len(samples),
        "_out": out,
    }


# ---------------------------------------------------------------------------
# digests / records
# ---------------------------------------------------------------------------
def digest(value) -> Dict[str, Any]:
    """Shape + value fingerprint of a result tensor (cross-process comparison)."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().to("cpu", torch.float32).numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.ascontiguousarray(array)
    nan = int(np.isnan(array).sum())
    posinf = int(np.isposinf(array).sum())
    neginf = int(np.isneginf(array).sum())
    safe = np.where(np.isfinite(array), array, np.float32(0.0))
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest()[:32],
        "finite": int(np.isfinite(array).sum()),
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "abs_sum": float(np.abs(safe.astype(np.float64)).sum()),
    }


def public_timing(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the private result tensor before serializing."""
    out = dict(record)
    out.pop("_out", None)
    return out


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# route introspection
# ---------------------------------------------------------------------------
def route_decision(
    pop: int, rows: int, eval_mode: Optional[str], backend: Optional[str]
) -> Dict[str, Any]:
    """What ``Forest.forward``/``SR_fitness`` would do for this cell, per the policy.

    ``should_use_dense`` is the ONE policy function every wired call site asks,
    so this reports the real decision (not a re-derivation).  ``rows`` is the
    data-row count for the ``per_row=False`` workloads, ``1`` for the per-row
    one (which is never rows-gated).
    """
    from evogp.tree import dense_engine

    decision = {
        "eval_mode": dense_engine.resolve_eval_mode(eval_mode),
        "backend": dense_engine.resolve_backend_name(backend, None),
    }
    # The wired call sites pass pop_size only; evaluate_dense/sr_fitness_dense
    # enforce the rows half, so both are reported.
    decision["should_use_dense_pop_only"] = bool(
        dense_engine.should_use_dense(pop, eval_mode=eval_mode, backend=backend)
    )
    decision["should_use_dense_with_rows"] = bool(
        dense_engine.should_use_dense(pop, eval_mode=eval_mode, backend=backend, rows=rows)
    )
    decision["would_use_dense"] = decision["should_use_dense_pop_only"] and not bool(
        dense_engine.auto_rows_decline(rows, eval_mode=decision["eval_mode"])
    )
    return decision
