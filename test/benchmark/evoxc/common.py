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
    # one merged graph per population (structure multiset), via evoxc.batch_functions
    "evoxc-batch-iree": {"eval_mode": "batch", "backend": "iree", "family": "evoxc"},
    # the same route with flat (unbanded) merging: O(N^2) in the population
    "evoxc-batch-iree-flat": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_BATCH_BANDS": "flat"},
    },
    # dense with its trace cap pinned to the CELL's tree length: the cap is a
    # ``{tree_len}`` template resolved by ``arms.apply_arm_env(arm, cell)``, so on a
    # main-grid cell (L <= 64) this is exactly ``evoxc-dense-iree``, and on a
    # long-tree cell it is the same route with the cap raised so dense actually
    # serves the population instead of declining to ``specialized``.
    "evoxc-dense-iree-raised": {
        "eval_mode": "dense",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_DENSE_MAX_TREE_LEN": "{tree_len}"},
    },
}
TORCH_ARMS: Dict[str, Dict[str, Any]] = {
    "torch-per-tree": {"impl": "per_tree", "family": "torch"},
    "torch-batched": {"impl": "batched", "family": "torch"},
}
ALL_ARMS: List[str] = list(TORCH_ARMS) + list(EVOXC_ARMS)

#: The knob table of the batch route (``EVOGP_EVAL_MODE=batch``), the ONE place
#: the harness names them: ``(spec key, env var, documented default, candidate
#: resolver attribute names on ``evogp.tree.batch_engine``)``.  ``batch_policy()``
#: is the only reader, so an arm's ``knobs`` mapping, an exported variable and the
#: engine's own default are never spelled out twice.
#:
#: ``EVOGP_BATCH_BANDS`` accepts an explicit band count ``K`` or ``none``/``flat``
#: (one unbanded merge, O(N^2) in the population -- the ``-flat`` arm times it);
#: ``EVOGP_BATCH_ALIGNMENT`` defaults to ``variant`` (band variants are aligned
#: before the merge instead of being padded per call).
BATCH_KNOBS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    # (spec key, env var, documented default, candidate resolver attribute names)
    ("band_size", "EVOGP_BATCH_BAND_SIZE", "32", ("resolve_band_size",)),
    ("bands", "EVOGP_BATCH_BANDS", "default", ("resolve_bands", "resolve_band_count")),
    ("switch_ops", "EVOGP_BATCH_SWITCH_OPS", "off", ("resolve_switch_ops", "uses_switch_ops")),
    ("alignment", "EVOGP_BATCH_ALIGNMENT", "variant", ("resolve_alignment",)),
    ("cache_size", "EVOGP_BATCH_CACHE_SIZE", "unset", ("resolve_cache_size",)),
    ("min_pop", "EVOGP_BATCH_MIN_POP", "64", ("resolve_min_pop",)),
)
#: Every env var the table above owns -- the arms clear all of them per process.
BATCH_ENV_VARS: Tuple[str, ...] = tuple(entry[1] for entry in BATCH_KNOBS)
#: The DENSE route's trace-cap knob (``dense_engine.MAX_TREE_LEN_ENV``, single-sourced
#: here so the harness names it once): the longest trace ``L = min(cap,
#: max_tree_len)`` the dense evaluator will build, default 64 -- above it the route
#: declines with ``DenseUnsupported`` and the public API falls back to specialized.
DENSE_ENV_VARS: Tuple[str, ...] = ("EVOGP_DENSE_MAX_TREE_LEN",)
#: Every env var the arms clear before applying their own knob spec, so a route knob
#: can never leak between arms (``BATCH_ENV_VARS`` + ``DENSE_ENV_VARS``).
ROUTE_ENV_VARS: Tuple[str, ...] = BATCH_ENV_VARS + DENSE_ENV_VARS
#: The counter names the batch route's compile misses can live under, in preference
#: order.  ``evogp.tree.batch_engine.batch_cache_stats()`` names its two front tables
#: explicitly: ``batch_exec_*`` describes ``BATCH_EXEC_MEMO`` -- the real compile
#: cache, since a hit never reaches ``evoxc.lower`` / ``compile`` / ``load`` -- and
#: ``batch_plan_*`` describes ``BATCH_PLAN_MEMO`` (host decode + traces + merge).
#: A bare ``misses`` is this harness's own older spelling: a fallback only.
BATCH_EXEC_MISS_KEYS: Tuple[str, ...] = ("batch_exec_misses", "misses")
#: The plan table's counter: a plan miss re-traces and re-merges, it never recompiles.
BATCH_PLAN_MISS_KEYS: Tuple[str, ...] = ("batch_plan_misses",)
#: The exec-table counter name -- the one a record's ``stats_keys`` must carry.
BATCH_EXEC_MISS_KEY = BATCH_EXEC_MISS_KEYS[0]
#: Location of the batch engine relative to the ``src`` directory (import-free check).
BATCH_MODULE_REL = os.path.join("evogp", "tree", "batch_engine.py")


def is_batch_arm(arm: str) -> bool:
    """Is this arm's route the batch engine (one merged graph per population)?

    One definition for ``worker.py`` / ``check.py`` / the economics script: an arm
    is a batch arm exactly when its registry entry asks for ``EVOGP_EVAL_MODE=batch``,
    so a future batch variant (another band policy) is picked up with no extra rule.
    """
    return (EVOXC_ARMS.get(arm) or {}).get("eval_mode") == "batch"

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
#: the long-tree column's tree lengths -- both above the dense route's default trace
#: cap (``dense_engine.DEFAULT_MAX_TREE_LEN`` = 64, ``EVOGP_DENSE_MAX_TREE_LEN``)
LONG_TREE_LENS = (128, 256)
INPUT_LENS = (8, 32)
MAIN_ROWS = 1024
#: the second data-row value ("one N=64 column"): L=64/input_len=32 at every pop
N64_ROWS = 64


def cell(
    pop: int, tree_len: int, input_len: int, rows: int, max_layer_cnt: int = 5
) -> Dict[str, Any]:
    """One cell: the population shape plus the generation depth cap.

    ``tree_len`` is the row stride (the descriptor's ``max_tree_len``) and
    ``max_layer_cnt`` is the depth cap that governs how many nodes the generated
    trees actually carry (a full tree is ``2 ** d - 1`` nodes).  The id keeps the
    historical spelling for the default cap (``max_layer_cnt == 5``) and gains a
    ``-d<k>`` suffix otherwise, so every existing cell id stays byte-identical.
    """
    suffix = "" if int(max_layer_cnt) == 5 else f"-d{int(max_layer_cnt)}"
    return {
        "id": f"p{pop}-L{tree_len}-i{input_len}-N{rows}{suffix}",
        "pop": pop,
        "tree_len": tree_len,
        "input_len": input_len,
        "rows": rows,
        "max_layer_cnt": int(max_layer_cnt),
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


def long_tree_cells() -> List[Dict[str, Any]]:
    """The LONG-TREE column at ``N = 1024`` (``--cells long``): six cells.

    The column has TWO axes and the cells separate them, because in this library
    they are different quantities:

    * ``tree_len`` is the descriptor's ``max_tree_len`` -- the row stride of the
      tree tensors and the length the dense route traces.  The dense route
      declines every population whose longest tree exceeds
      ``EVOGP_DENSE_MAX_TREE_LEN`` (default 64), so above that cap the public API
      silently falls back to the per-tree ``specialized`` route (one compile per
      distinct structure).  A cell therefore has to be priced with the cap both at
      its default (``evoxc-dense-iree``, measuring the fallback) and raised to the
      cell's own tree length (``evoxc-dense-iree-raised``).
    * ``max_layer_cnt`` is what actually decides how many nodes a generated tree
      carries (a full tree is ``2 ** d - 1`` nodes), so it -- not ``tree_len`` --
      governs the evaluation work.  Measured on this box with the
      :func:`build_workload` recipe (seed 0, pop 64), the node count is identical
      at L=32, L=64 and L=128 for one depth cap and grows with the cap:
      ``d=5`` mean 15.8 / max 29 nodes, ``d=6`` mean 27.1 / max 57, ``d=7``
      mean 35.5 / max 101, ``d=8`` mean 69.9 / max 187.

    Hence the pair of cells per tree length: the cap-5 cell is the stock
    generation with only the row stride made long, and each ``-d<k>`` sibling
    raises the depth cap (the real node-count axis) at that same stride.

    ``tree_len`` 128/256 and the caps up to 8 are legal descriptor settings
    (``MAX_STACK`` = 1024 is not a constraint at these lengths, and
    ``2 ** d - 1 <= tree_len`` holds for every cell below), and the population
    comes from the same :func:`build_workload` recipe as every other column --
    ``torch.manual_seed(0)`` + ``Forest.random_generate`` -- so only the shape is
    new.
    """
    return [
        cell(256, 128, 32, MAIN_ROWS, 5),
        cell(256, 128, 32, MAIN_ROWS, 6),
        cell(256, 128, 32, MAIN_ROWS, 7),
        cell(1000, 128, 32, MAIN_ROWS, 7),
        cell(256, 256, 32, MAIN_ROWS, 5),
        cell(256, 256, 32, MAIN_ROWS, 8),
    ]


def parse_cell(text: str) -> Dict[str, Any]:
    """``p256-L64-i32-N1024`` -> the cell dict (``-d7`` sets the depth cap)."""
    parts = dict(part.split("=", 1) if "=" in part else (part[0], part[1:]) for part in text.split("-"))
    try:
        return cell(
            int(parts["p"]),
            int(parts["L"]),
            int(parts["i"]),
            int(parts["N"]),
            int(parts.get("d", 5)),
        )
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


def build_workload(
    cell_spec: Dict[str, Any],
    seed: int = 0,
    using_funcs=None,
    max_layer_cnt: Optional[int] = None,
) -> Dict[str, Any]:
    """Deterministically build the population and data of one cell.

    ``torch.manual_seed(seed)`` + ``Forest.random_generate`` is the documented
    reproducibility recipe, so every arm in every process sees the same trees.
    The generation depth cap comes from the CELL (``max_layer_cnt``, default 5)
    unless the caller passes one explicitly, so a cell's id and its population
    always agree on the node budget.
    """
    import torch
    from evogp.tree import Forest

    if max_layer_cnt is None:
        max_layer_cnt = int(cell_spec.get("max_layer_cnt", 5))
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


def population_signature(forest) -> Dict[str, Any]:
    """Structure-multiset fingerprint of a population (the batch route's cache key).

    The merged batch graph is keyed on the *multiset* of tree structures, not on
    the row order, so the per-row hashes are collected into a set and sorted
    before the population hash: a permuted population must report the same
    ``signature``.  ``distinct_structures == 1`` means every row is the same
    structure (elites and duplicates dominate), which is where one merged graph
    wins by the widest margin; ``distinct_structures == pop_size`` means a fresh
    population of all-distinct trees, i.e. the batch route's worst case.

    Read through ``Forest.batch_*`` only: no torch import, and every tensor is
    moved to CPU with ``np.ascontiguousarray`` first (they live on the device and
    are not guaranteed contiguous after generation/mutation).
    """
    import numpy as np

    value = np.ascontiguousarray(forest.batch_node_value.detach().to("cpu").numpy())
    node_type = np.ascontiguousarray(forest.batch_node_type.detach().to("cpu").numpy())
    subtree = np.ascontiguousarray(forest.batch_subtree_size.detach().to("cpu").numpy())

    rows = set()
    for row in range(int(forest.pop_size)):
        length = int(subtree[row, 0])
        payload = (
            np.ascontiguousarray(value[row, :length]).tobytes()
            + np.ascontiguousarray(node_type[row, :length]).tobytes()
            + np.ascontiguousarray(subtree[row, :length]).tobytes()
        )
        rows.add(hashlib.sha256(payload).hexdigest())
    return {
        "distinct_structures": len(rows),
        "signature": hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()[:16],
        "pop_size": int(forest.pop_size),
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
def _requested_eval_mode(eval_mode: Optional[str]) -> Optional[str]:
    """The route name the caller actually asked for, env fallback included.

    Stated once so ``route_decision`` can special-case a route before handing the
    request to the dense policy: an explicit argument wins over
    ``EVOGP_EVAL_MODE``, and neither case nor surrounding whitespace matters.
    ``None`` means "no route pinned" (the library default).
    """
    raw = eval_mode if eval_mode is not None else os.environ.get("EVOGP_EVAL_MODE")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _source_root_candidate() -> Optional[str]:
    """The ``src`` directory to inspect, resolved WITHOUT importing ``evogp``.

    ``src_root()`` is authoritative when ``evogp`` is already imported (that is
    the ``--tree-root`` checkout under measurement); before that import -- e.g. in
    ``check.py``, which is stdlib-only -- the harness's own checkout is found by
    walking up from this file for ``src/evogp``.  ``None`` when neither exists.
    """
    if "evogp" in sys.modules:
        try:
            return src_root()
        except Exception:  # noqa: BLE001 - a broken import must not raise here
            pass
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(path, "src", "evogp")):
            return os.path.join(path, "src")
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def batch_engine_present(src: Optional[str] = None) -> bool:
    """Does the tree under test ship ``evogp/tree/batch_engine.py``?  (No import.)

    Mirrors ``cold_start.artifact_feature_present``: a path check keeps the answer
    available before ``evogp`` is imported (a snapshot or a pre-batch tree simply
    has no such file), and this function never imports ``evogp`` and never raises.
    """
    if src is None:
        src = _source_root_candidate()
    if not src:
        return False
    try:
        return os.path.isfile(os.path.join(src, BATCH_MODULE_REL))
    except Exception:  # noqa: BLE001 - a bad path is "absent", never a crash
        return False


def _registered_knob_value(env_name: str) -> Optional[str]:
    """The value an arm of the registry would write for ``env_name`` (or None).

    Lets ``batch_policy`` label a variable as arm-supplied instead of merely
    inherited: arms write their ``knobs`` mapping into ``os.environ`` before the
    engine is imported, so the registry is the only record of who set it.
    """
    for spec in EVOXC_ARMS.values():
        value = (spec.get("knobs") or {}).get(env_name)
        if value is not None:
            return str(value)
    return None


def _resolve_batch_knob(
    module, spec_key: str, env_name: str, default: str, candidates: Tuple[str, ...], errors: Dict[str, str]
) -> Tuple[str, str]:
    """``(value, source)`` for one batch knob, in the documented precedence order.

    arm-supplied env value > inherited env value > the engine's own resolver >
    the documented default.  A failing resolver is recorded in ``errors`` and the
    next candidate (then the default) is used, so a policy report is never lost to
    one broken accessor.
    """
    env_value = os.environ.get(env_name)
    if env_value is not None:
        arm_value = _registered_knob_value(env_name)
        return env_value, "arm" if arm_value == env_value else "env"
    if module is not None:
        for attr in candidates:
            resolver = getattr(module, attr, None)
            if callable(resolver):
                try:
                    return str(resolver()), "module"
                except Exception as exc:  # noqa: BLE001 - report, then try/fall back
                    errors[spec_key] = repr(exc)
    return default, "harness-default"


def batch_policy(src: Optional[str] = None) -> Dict[str, Any]:
    """Everything a record must state about the batch route as THIS process sees it.

    Import-light by construction: ``evogp.tree.dense_engine`` is consulted only for
    its declared ``EVAL_MODES`` (``mode_supported`` tells whether today's tree even
    accepts ``EVOGP_EVAL_MODE=batch``), and ``evogp.tree.batch_engine`` only for the
    knob resolvers named in :data:`BATCH_KNOBS`.  Both imports are guarded, so a
    GPU-free / evogp-free caller (``check.py``, ``summarize.py``) still gets a full
    report with ``module_present: False`` instead of an exception.
    """
    present = batch_engine_present(src)
    resolved_src = src if src is not None else _source_root_candidate()
    module_path = os.path.join(resolved_src, BATCH_MODULE_REL) if resolved_src else None
    policy: Dict[str, Any] = {
        "module_present": present,
        "module": module_path,
        "eval_mode": "batch",
        "backend": "iree with numpy fallback",
        "declared_modes": None,
        "mode_supported": False,
    }
    errors: Dict[str, str] = {}

    # Does the installed policy know the route at all?  Its EVAL_MODES is the
    # authority; on today's tree ("dense", "specialized", "auto") it does not.
    try:
        from evogp.tree import dense_engine

        declared = list(getattr(dense_engine, "EVAL_MODES", ()) or ())
        policy["declared_modes"] = declared or None
        policy["mode_supported"] = "batch" in declared
    except Exception as exc:  # noqa: BLE001 - absence is a reportable state
        policy["declared_modes_error"] = repr(exc)

    module = None
    try:
        from evogp.tree import batch_engine as module
    except Exception as exc:  # noqa: BLE001 - the route can be absent
        policy["module_error"] = repr(exc)

    knobs: Dict[str, Optional[str]] = {}
    sources: Dict[str, str] = {}
    values: Dict[str, str] = {}
    defaults = {entry[0]: entry[2] for entry in BATCH_KNOBS}
    for spec_key, env_name, default, candidates in BATCH_KNOBS:
        value, source = _resolve_batch_knob(module, spec_key, env_name, default, candidates, errors)
        values[spec_key] = value
        sources[spec_key] = source
        knobs[env_name] = os.environ.get(env_name)
    if errors:
        policy["resolver_errors"] = errors

    # The two numeric knobs are coerce-only: a record should carry ints, and a
    # bad value falls back to the documented default with the reason recorded.
    for spec_key in ("min_pop", "band_size"):
        try:
            policy[spec_key] = int(values[spec_key])
        except (TypeError, ValueError) as exc:
            policy.setdefault("resolver_errors", {})[spec_key] = repr(exc)
            policy[spec_key] = int(defaults[spec_key])
    for spec_key in ("bands", "switch_ops", "alignment", "cache_size"):
        policy[spec_key] = str(values[spec_key])
    policy["knobs"] = knobs
    policy["knobs_source"] = sources
    return policy


def pop_min_declines(pop: int, min_pop: int) -> Optional[str]:
    """The batch route's documented decline rule: reason string below the floor.

    Contract agreed with the engine, which owns the real decision: below
    ``EVOGP_BATCH_MIN_POP`` (default 64) a merged graph cannot amortize its build,
    so the route declines to ``specialized``.  This mirrors that decision for the
    route summary; it never decides anything itself.
    """
    if int(pop) < int(min_pop):
        return (
            f"pop {int(pop)} < EVOGP_BATCH_MIN_POP {int(min_pop)}: one merged graph "
            "cannot amortize its build, so the route declines to specialized"
        )
    return None


def batch_route_decision(pop: int, rows: Optional[int] = None) -> Dict[str, Any]:
    """The batch analogue of :func:`route_decision`, in the same shape family.

    ``summarize.route_text`` reads ``eval_mode`` / ``backend`` / ``would_use_dense``
    from it, so those keys are always present (the batch route is never dense and
    its ``kind`` says so explicitly).  ``would_use_batch`` folds the three gates
    together -- module present, mode declared by the tree, population above the
    floor -- and ``rows`` is echoed only because every other route decision takes
    it: the batch route is not rows-gated.
    """
    policy = batch_policy()
    decline = pop_min_declines(pop, policy["min_pop"])
    would_use = bool(policy["module_present"] and policy["mode_supported"] and decline is None)
    if not policy["module_present"]:
        state = "batch_engine.py absent on this tree"
    elif not policy["mode_supported"]:
        state = "the tree's EVAL_MODES does not declare 'batch'"
    elif decline is not None:
        state = "declines to specialized below the population floor"
    else:
        state = "merged graph per population structure multiset"
    return {
        "eval_mode": "batch",
        "backend": "iree",
        "kind": "batch",
        "would_use_dense": False,
        "would_use_batch": would_use,
        "declines_to_specialized": decline is not None,
        "decline_reason": decline,
        "min_pop": policy["min_pop"],
        "band_size": policy["band_size"],
        "bands": policy["bands"],
        "alignment": policy["alignment"],
        "switch_ops": policy["switch_ops"],
        "module_present": policy["module_present"],
        "mode_supported": policy["mode_supported"],
        "rows": rows,
        "note": f"batch/iree: {state}",
    }


def route_decision(
    pop: int, rows: int, eval_mode: Optional[str], backend: Optional[str]
) -> Dict[str, Any]:
    """What ``Forest.forward``/``SR_fitness`` would do for this cell, per the policy.

    ``should_use_dense`` is the ONE policy function every wired call site asks,
    so this reports the real decision (not a re-derivation).  ``rows`` is the
    data-row count for the ``per_row=False`` workloads, ``1`` for the per-row
    one (which is never rows-gated).
    """
    # The batch route is intercepted BEFORE the dense policy is imported: on the
    # current tree dense_engine.EVAL_MODES == ("dense", "specialized", "auto"), so
    # resolve_eval_mode("batch") raises ValueError and the batch decision must come
    # from its own module (batch_route_decision), never from that call.
    if _requested_eval_mode(eval_mode) == "batch":
        return batch_route_decision(pop, rows)

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
