#!/usr/bin/env python
"""``make_r2_fixtures.py`` -- build the R2 200-tree cross-side evaluation fixture.

The R2 round of the benchmark evaluates **one fixed forest of 200 trees on both
engines** (OLD = the legacy custom-CUDA checkout at ``b613503``, NEW = this repo
at HEAD with the ETL "true jit" engine).  The committed ``test/goldens/forest_random.npz``
fixture only holds 64 trees, so this script generates a 200-tree one with exactly
the same npz schema and the SPEC.md W1 descriptor, which lets
``bench_gp.py --task eval64 --fixture <this file>`` run unchanged on either side
(and keeps ``--data-rows`` tiling usable).

Schema written (identical to the ``forest_random`` capture in ``test/goldens/``)::

    forest__node_value    (pop, max_tree_len)  float32   # Forest ctor arg 1
    forest__node_type     (pop, max_tree_len)  int16     # Forest ctor arg 2
    forest__subtree_size  (pop, max_tree_len)  int16     # Forest ctor arg 3
    x                     (rows, 2)            float32   # copied from the source fixture
    labels                (rows, 1)            float32   # copied from the source fixture

The three ``forest__*`` arrays are plain :class:`evogp.tree.Forest` constructor
inputs (float32 / int16 / int16, ``batch_node_value`` / ``batch_node_type`` /
``batch_subtree_size``); ``x``/``labels`` are copied verbatim from the source
fixture (one shared input/label set, so the comparison is RNG-independent).
``np.savez`` (uncompressed, fixed 1980 zip timestamps) keeps the file
byte-for-byte reproducible for a fixed seed/pop/tree-len.

Regeneration (from the repository root, GPU needed -- the descriptor builds its
tables on ``"cuda"``)::

    PYTHONPATH=src python test/benchmark/make_r2_fixtures.py

Use ``--out`` / ``--pop`` / ``--tree-len`` / ``--seed`` / ``--data-from`` to
overwrite any of those defaults (``--out`` defaults to the committed fixture
path, so the bare command above regenerates it in place).  ``--verify`` reloads
the written file, rebuilds the ``Forest`` from it and runs one ``SR_fitness``
call through the public API (a decode/eval smoke check).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

# The repository's recommended ETL setting; must be fixed before ``evogp`` (and
# therefore ``etl``) is imported.  Fixture generation builds no ETL graph, so this
# only keeps the process consistent with the benchmark's environment contract.
os.environ.setdefault("ETL_DISABLE_LOCATIONS", "1")

# --- bootstrapping ----------------------------------------------------------
# ``src/`` is not installed in every environment this script runs in, so make the
# sibling source tree importable when it is present (never overrides PYTHONPATH).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.append(_SRC)

from evogp.tree import Forest, GenerateDescriptor  # noqa: E402  (after sys.path fix)

DEFAULT_OUT = os.path.join(_HERE, "fixtures", "forest200_tree64_random.npz")
DEFAULT_DATA_FROM = os.path.join(_REPO_ROOT, "test", "goldens", "forest_random.npz")

DEFAULT_POP = 200
DEFAULT_TREE_LEN = 64
DEFAULT_SEED = 0

#: descriptor parameters from ``SPEC.md`` §W1 / ``bench_gp.py::w1_descriptor``
#: (replicated here on purpose: this script must not import the harness).
INPUT_LEN = 2
OUTPUT_LEN = 1
DESCRIPTOR_KWARGS = dict(
    input_len=INPUT_LEN,
    output_len=OUTPUT_LEN,
    const_prob=0.5,
    out_prob=0.5,
    using_funcs=["+", "-", "*", "/"],
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5,
)

#: ``evogp.tree.utils.NType`` base kinds (``VAR``, ``CONST``, ``UFUNC``, ``BFUNC``,
#: ``TFUNC``); ``NType.OUT_NODE`` (0x80) may be OR-ed on top.
NODE_KIND_NAMES = {0: "var", 1: "const", 2: "ufunc", 3: "bfunc", 4: "tfunc"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"npz path to write (default: {DEFAULT_OUT})")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP, help="population size (default: %(default)s)")
    parser.add_argument(
        "--tree-len",
        type=int,
        default=DEFAULT_TREE_LEN,
        help="descriptor max_tree_len / array width (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch RNG seed (default: %(default)s)")
    parser.add_argument(
        "--data-from",
        default=DEFAULT_DATA_FROM,
        help=f"npz to copy x/labels from (default: {DEFAULT_DATA_FROM})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="reload the written file, rebuild the Forest and run one SR_fitness call",
    )
    args = parser.parse_args(argv)
    if args.pop < 1:
        parser.error("--pop must be >= 1")
    if args.tree_len < 1:
        parser.error("--tree-len must be >= 1")
    return args


def r2_descriptor(tree_len):
    """SPEC.md §W1 descriptor (``GenerateDescriptor`` builds its tables on cuda)."""
    return GenerateDescriptor(max_tree_len=tree_len, **DESCRIPTOR_KWARGS)


def generate_forest(pop_size, tree_len, seed):
    """Seeded ``Forest.random_generate``: ``(descriptor, forest)``."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    descriptor = r2_descriptor(tree_len)
    forest = Forest.random_generate(pop_size=pop_size, descriptor=descriptor)
    return descriptor, forest


def load_data_arrays(path, input_len, output_len):
    """``(x, labels)`` copied from ``path`` as contiguous float32 arrays."""
    with np.load(path) as data:
        missing = [k for k in ("x", "labels") if k not in data.files]
        assert not missing, f"{path} lacks {missing}; keys: {list(data.files)}"
        x = np.ascontiguousarray(data["x"], dtype=np.float32)
        labels = np.ascontiguousarray(data["labels"], dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] == input_len, f"x should be (rows, {input_len}), got {x.shape}"
    assert labels.ndim == 2 and labels.shape[1] == output_len, (
        f"labels should be (rows, {output_len}), got {labels.shape}"
    )
    assert labels.shape[0] == x.shape[0], (
        f"x and labels disagree on rows: {x.shape[0]} vs {labels.shape[0]}"
    )
    return x, labels


def forest_arrays(forest):
    """The three ``forest__*`` arrays in the committed fixture layout/dtypes."""
    return {
        "forest__node_value": np.ascontiguousarray(
            forest.batch_node_value.detach().cpu().numpy(), dtype=np.float32
        ),
        "forest__node_type": np.ascontiguousarray(
            forest.batch_node_type.detach().cpu().numpy(), dtype=np.int16
        ),
        "forest__subtree_size": np.ascontiguousarray(
            forest.batch_subtree_size.detach().cpu().numpy(), dtype=np.int16
        ),
    }


def tree_report(arrays, tree_len):
    """Structural sanity stats + hard checks on the generated rows."""
    node_value = arrays["forest__node_value"]
    node_type = arrays["forest__node_type"]
    subtree_size = arrays["forest__subtree_size"]
    lengths = subtree_size[:, 0].astype(np.int64)
    width = node_value.shape[1]

    assert width == tree_len, f"array width {width} != tree_len {tree_len}"
    assert node_value.shape == node_type.shape == subtree_size.shape
    assert node_value.dtype == np.float32, node_value.dtype
    assert node_type.dtype == np.int16, node_type.dtype
    assert subtree_size.dtype == np.int16, subtree_size.dtype
    assert lengths.min() >= 1, f"degenerate tree: subtree_size[:, 0] = {lengths.min()}"
    assert lengths.max() <= width, f"tree longer than the array: {lengths.max()} > {width}"

    # every row must be the documented prefix encoding: zeros after the tree length
    used = np.arange(width)[None, :] < lengths[:, None]
    for key, arr in (
        ("forest__node_value", node_value),
        ("forest__node_type", node_type),
        ("forest__subtree_size", subtree_size),
    ):
        assert not np.any(arr[~used]), f"{key}: non-zero bytes in the dead tail (index >= subtree_size[:, 0])"

    # node kinds: VAR/CONST/UFUNC/BFUNC/TFUNC, optional OUT_NODE flag (NType in tree/utils.py)
    raw_types = node_type[used]
    assert not np.any(raw_types & ~np.int16(0x7F | 0x80)), (
        f"unexpected bits in node_type: {np.unique(raw_types & ~np.int16(0x7F | 0x80))}"
    )
    base = (raw_types & 0x7F).astype(np.int64)
    assert base.max() <= max(NODE_KIND_NAMES), f"unknown node kind ids: {np.unique(base)}"
    kinds_u, kinds_c = np.unique(base, return_counts=True)
    kinds = {NODE_KIND_NAMES[int(k)]: int(c) for k, c in zip(kinds_u, kinds_c)}
    assert np.isfinite(node_value[used]).all(), "non-finite node_value inside a tree"

    return {
        "pop": int(node_value.shape[0]),
        "width": int(width),
        "tree_len_min": int(lengths.min()),
        "tree_len_median": float(np.median(lengths)),
        "tree_len_max": int(lengths.max()),
        "unique_tree_lens": int(np.unique(lengths).size),
        "unique_rows": int(np.unique(node_value, axis=0).shape[0]),
        "node_kinds": kinds,
        "out_node_tagged": int(((raw_types & 0x80) != 0).sum()),
        "node_value_all_finite": True,
        "dead_tail_zeros": True,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays):
    """Order-independent sha256 over key names + canonical (C-order) bytes."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def verify_fixture(path, input_len, output_len):
    """Decode smoke check through the public API: Forest(...) + one SR_fitness call.

    NaN/±inf fitness values are expected for a random population (``/`` by an
    expression that vanishes on some data row), exactly as in the legacy
    ``sr_fitness`` capture (3 of 64 trees NaN at ``test/goldens/``); the count is
    reported instead of being treated as an error.
    """
    with np.load(path) as data:
        loaded = {k: data[k] for k in data.files}
    x = torch.from_numpy(np.ascontiguousarray(loaded["x"], dtype=np.float32)).to("cuda")
    labels = torch.from_numpy(np.ascontiguousarray(loaded["labels"], dtype=np.float32)).to("cuda")
    forest = Forest(
        input_len=input_len,
        output_len=output_len,
        batch_node_value=torch.from_numpy(np.ascontiguousarray(loaded["forest__node_value"])).to("cuda"),
        batch_node_type=torch.from_numpy(np.ascontiguousarray(loaded["forest__node_type"])).to("cuda"),
        batch_subtree_size=torch.from_numpy(np.ascontiguousarray(loaded["forest__subtree_size"])).to("cuda"),
    )
    fitness = forest.SR_fitness(x, labels)
    finite = torch.isfinite(fitness)
    n_finite = int(finite.sum().item())
    return {
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "n_data_rows": int(x.shape[0]),
        "fitness_nan_or_inf_count": int(fitness.numel() - n_finite),
        "fitness_finite_mean": float(fitness[finite].mean().item()) if n_finite else None,
        "fitness_finite_min": float(fitness[finite].min().item()) if n_finite else None,
        "fitness_finite_max": float(fitness[finite].max().item()) if n_finite else None,
    }


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible (the descriptor builds its tables on 'cuda')", file=sys.stderr)
        return 2

    print(f"[r2fixture] evogp={os.path.dirname(os.path.abspath(__import__('evogp').__file__))}")
    print(f"[r2fixture] torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
    print(
        f"[r2fixture] pop={args.pop} tree_len={args.tree_len} seed={args.seed} "
        f"data_from={os.path.abspath(args.data_from)} out={os.path.abspath(args.out)}"
    )

    x, labels = load_data_arrays(args.data_from, INPUT_LEN, OUTPUT_LEN)
    descriptor, forest = generate_forest(args.pop, args.tree_len, args.seed)
    assert descriptor.input_len == INPUT_LEN and descriptor.output_len == OUTPUT_LEN

    arrays = forest_arrays(forest)
    arrays["x"] = x
    arrays["labels"] = labels

    report = tree_report(arrays, tree_len=descriptor.max_tree_len)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    # np.savez (uncompressed) writes fixed 1980 zip timestamps: byte-for-byte stable.
    np.savez(args.out, **arrays)

    print("[r2fixture] arrays written:")
    for key in arrays:
        arr = arrays[key]
        print(f"[r2fixture]   {key:22s} shape={arr.shape} dtype={arr.dtype} C={arr.flags['C_CONTIGUOUS']}")
    print(f"[r2fixture] tree stats: {report}")
    print(f"[r2fixture] sha256(file)  = {sha256_file(args.out)}")
    print(f"[r2fixture] sha256(arrays)= {array_digest(arrays)}")

    if args.verify:
        print(f"[r2fixture] verify: {verify_fixture(args.out, INPUT_LEN, OUTPUT_LEN)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
