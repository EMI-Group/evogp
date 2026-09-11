"""Descriptor-driven random tree generation in plain vectorized PyTorch.

This module is the pure-torch replacement of the legacy compiled extension's
``tree_generate`` kernel (whose semantics are distilled in ``ENGINE_SPEC.md``
§6): it draws a
fresh population of GP trees from a :class:`GenerateDescriptor` using nothing
but batched tensor operations, so it does not rely on the compiled extension
and runs on any torch device (CUDA/HIP GPU or CPU).

Public interface
----------------
``generate_population(pop_size, descriptor, device=None)`` returns the raw
``(batch_node_value, batch_node_type, batch_subtree_size)`` triple -- exactly
the three tensors ``Forest`` stores (float32 / int16 / int16, shaped
``[pop_size, max_tree_len]``).

How the legacy kernel decides (reproduced here, ``ENGINE_SPEC.md`` §6)
---------------------------------------------------------------------
The kernel walks one tree at a time with an explicit stack, in prefix
(pre-order) order. For every node it draws, with ``r ~ U[0, 1)``:

* ``r >= depth2leaf_probs[depth]`` -> *function* node, otherwise a *leaf*;
* function id ``k`` from the ``roulette_funcs`` cdf: ``k`` is the number of cdf
  entries ``<= r`` (the kernel scans ``i`` downwards and takes ``1 + max{i :
  cdf[i] <= r}``), so ``k == searchsorted(cdf, r, right=True)``;
* arity follows from the id: ``0`` (``if``) -> 3, ``1..13`` -> 2, ``14..28`` -> 1;
* multi-output only (``output_len > 1``): with probability ``out_prob`` the node
  is tagged as an out-node (``node_type |= OUT_NODE``) and its value becomes the
  bit-reinterpretation of ``{function, outIndex}`` (function in the low 16 bits,
  the sampled output index ``engine() % output_len`` in the high 16 bits);
* leaves: with probability ``const_prob`` a constant drawn uniformly from
  ``const_samples``, otherwise a variable index ``engine() % input_len``;
* ``subtree_size[i] = 1 + sum(children sizes)`` and ``subtree_size[0]`` is the
  tree length.

Vectorization strategy
----------------------
Instead of one thread per tree following the DFS, the whole population is grown
**level by level**: one grid cell per live node, all nodes of the same depth are
decided with batched ops, and their children form the next level. A node's depth
-- and therefore every probability it is drawn with -- does not depend on the
traversal order, so the level-wise variant is *distributionally* identical to
the DFS kernel (the RNG stream itself differs by construction; the kernel seeds
a private taus88 per tree while this module consumes torch's global RNG, which
makes ``torch.manual_seed``/``torch.cuda.manual_seed`` reproducible).

Level-wise cells are in breadth-first order while the rest of the library
requires the prefix (pre-order) encoding, so two level-wise passes over the
(bounded, ``MAX_FULL_DEPTH`` at most) level list finish the job:

1. bottom-up: ``subtree_size`` of every cell from its children's sizes;
2. top-down: the prefix index of every cell, i.e. ``parent index + 1 + sizes of
   the preceding siblings`` (children of one parent occupy consecutive cells, so
   the needed sibling sizes are plain gathers; the result is scattered into the
   output rows with unique indices).

Only three Python loops are involved: over levels (<= ``MAX_FULL_DEPTH + 1``),
over the three possible child ranks, and over the small level list. No Python
loop is ever taken over individuals or nodes.

Documented behavior / deviations from the kernel
------------------------------------------------
* ``device=None`` keeps the descriptor's parameter tensors' device (today always
  ``"cuda"``, which is the historical behavior); any explicit device works, e.g.
  ``"cpu"``. Parameters are copied to the target device, the outputs are
  allocated on it.
* Depths ``>= len(depth2leaf_probs)`` are treated as leaves, so the schedule is
  never indexed out of bounds. The kernel reads ``leafProbs[depth]`` past the end
  of its local array in that case (an unvalidated descriptor gets a silent GPU
  buffer overrun instead); here the node is deterministically a leaf.
* If a node that still has to grow children would make the tree exceed
  ``max_tree_len``, a :class:`RuntimeError` is raised. The kernel never
  bounds-checks ``gp[MAX_STACK]`` and silently corrupts memory instead. This is a
  descriptor validation problem: ``check_tree_length`` already guarantees
  ``max_tree_len >= (max_arity ** max_layer_cnt - 1) / (max_arity - 1)`` for a
  schedule with a ``1.0`` terminator at ``max_layer_cnt - 1``.
* Cells beyond ``subtree_size[:, 0]`` are deterministically zero here (the kernel
  leaves them as ``torch::empty`` garbage).
* The function id is clamped into ``[0, Func.END)`` for the (float-rounding only)
  case ``roulette_funcs[-1] < 1.0``, where the kernel would produce the invalid
  id ``Func.END``.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .utils import MAX_FULL_DEPTH, MAX_STACK, Func, NType


__all__ = ["generate_population"]


#: Maximum number of children any node of the library can have (``if``).
MAX_ARITY = 3

#: Mask of the function id stored in the low half of an out-node value.
_OUT_INDEX_SHIFT = 16


def _resolve_device(descriptor, device: Optional[object]) -> torch.device:
    """Device of the descriptor parameters, or the explicitly requested one."""
    if device is not None:
        return torch.device(device)
    for name in ("depth2leaf_probs", "roulette_funcs", "const_samples"):
        param = getattr(descriptor, name, None)
        if isinstance(param, Tensor):
            return param.device
    raise ValueError(
        "generate_population: cannot infer the device, the descriptor exposes "
        "none of depth2leaf_probs/roulette_funcs/const_samples"
    )


def _arity_from_type(node_type: Tensor) -> Tensor:
    """Number of children of every node, from its (possibly out-tagged) type."""
    base = (node_type & NType.TYPE_MASK).to(torch.int64)
    return torch.where(
        base <= NType.CONST, torch.zeros_like(base), base - 1
    )


def _level_active(ls: Tensor, le: Tensor, lo: int, hi: int, device) -> Tensor:
    """Mask of the cells holding a node of the level ``[ls, le)`` of each row.

    ``lo``/``hi`` bound the column window that contains the level of every
    individual of the population, so all cell-wise work of a level stays
    proportional to the widest level of the population instead of to
    ``max_tree_len``.
    """
    cols = torch.arange(lo, hi, dtype=torch.int64, device=device)
    return (cols.unsqueeze(0) >= ls.unsqueeze(1)) & (cols.unsqueeze(0) < le.unsqueeze(1))


def _child_base(
    node_type: Tensor,
    active: Tensor,
    le: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Column of the first child of every cell of a level.

    Children are appended in cells order and in rank order, so the first child
    of a cell is at ``level_end + #children of the preceding cells``. Returns
    ``(base_column, arity)``, both ``[pop_size, window]``; the arity of cells
    that do not hold a node of this level is zeroed out. ``base_column`` is only
    meaningful where ``arity > 0`` -- the capacity check of the growing pass
    guarantees that ``base_column + arity - 1 < max_tree_len`` there; the
    remaining cells are dropped by the ``arity > rank`` masks.
    """
    arity = _arity_from_type(node_type) * active.to(torch.int64)
    preceding = torch.cumsum(arity, dim=1) - arity
    return le.unsqueeze(1) + preceding, arity


def generate_population(
    pop_size: int,
    descriptor,
    device: Optional[object] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomly generate a population of GP trees from ``descriptor``.

    Reproduction of the legacy extension's ``tree_generate`` op in vectorized
    torch (see the module docstring for the exact decision structure and the
    documented deviations). Draws from torch's global RNG, therefore
    ``torch.manual_seed(seed)`` (+ ``torch.cuda.manual_seed(seed)`` for CUDA)
    reproduces the same population.

    Args:
        pop_size: number of trees to generate, a positive integer.
        descriptor: a :class:`GenerateDescriptor`-like object providing
            ``max_tree_len``, ``input_len``, ``output_len``, ``const_prob``,
            ``out_prob``, ``depth2leaf_probs``, ``roulette_funcs`` and
            ``const_samples``.
        device: torch device for the generated tensors. ``None`` (default) uses
            the device of the descriptor's parameter tensors.

    Returns:
        ``(batch_node_value, batch_node_type, batch_subtree_size)`` with shapes
        ``[pop_size, max_tree_len]`` and dtypes float32 / int16 / int16, every
        row being a valid prefix (pre-order) encoding of length
        ``batch_subtree_size[:, 0]`` and zeros afterwards.
    """
    assert (
        isinstance(pop_size, int) and pop_size > 0
    ), "pop_size should be a positive integer"

    dev = _resolve_device(descriptor, device)

    gp_len = int(descriptor.max_tree_len)
    input_len = int(descriptor.input_len)
    output_len = int(descriptor.output_len)
    const_prob = float(descriptor.const_prob)
    out_prob = float(descriptor.out_prob)

    assert (
        0 < gp_len <= MAX_STACK
    ), f"max_tree_len must be in range (0, {MAX_STACK}], but got {gp_len}"
    assert input_len > 0, f"input_len must be positive, but got {input_len}"
    assert output_len > 0, f"output_len must be positive, but got {output_len}"
    assert 0.0 <= const_prob <= 1.0, "const_prob should be in [0.0, 1.0]"
    assert 0.0 <= out_prob <= 1.0, "out_prob should be in [0.0, 1.0]"

    depth2leaf_probs = (
        descriptor.depth2leaf_probs.detach()
        .to(device=dev, dtype=torch.float32)
        .flatten()
    )
    roulette_funcs = (
        descriptor.roulette_funcs.detach().to(device=dev, dtype=torch.float32).flatten()
    )
    const_samples = (
        descriptor.const_samples.detach().to(device=dev, dtype=torch.float32).flatten()
    )
    assert depth2leaf_probs.numel() == MAX_FULL_DEPTH, (
        f"depth2leaf_probs should have {MAX_FULL_DEPTH} entries, "
        f"but got {depth2leaf_probs.numel()}"
    )
    assert (
        roulette_funcs.numel() == Func.END
    ), f"roulette_funcs should have {Func.END} entries, but got {roulette_funcs.numel()}"
    assert const_samples.numel() > 0, "const_samples should not be empty"

    n_probs = int(depth2leaf_probs.numel())
    # Depths at/after the first 1.0 entry can only produce leaves: the schedule
    # has a terminator there, which bounds the number of levels (and, together
    # with `check_tree_length`, the tree length). `n_probs` bounds it as well
    # since depths >= n_probs are always treated as leaves.
    terminator = (depth2leaf_probs >= 1.0).nonzero()
    last_level = n_probs if terminator.numel() == 0 else int(terminator[0, 0])
    last_level = min(last_level, n_probs)

    # ---- level-wise growth -------------------------------------------------
    # `value_grid` / `type_grid` are in breadth-first (level) order while they
    # are filled; they are reordered into the prefix encoding at the end.
    value_grid = torch.zeros((pop_size, gp_len), dtype=torch.float32, device=dev)
    type_grid = torch.zeros((pop_size, gp_len), dtype=torch.int16, device=dev)

    # cell range [ls, le) of the level currently processed, per individual
    ls = torch.zeros(pop_size, dtype=torch.int64, device=dev)
    le = torch.ones(pop_size, dtype=torch.int64, device=dev)
    lo, hi = 0, 1  # window covering the level of every individual
    levels: List[Tuple[Tensor, Tensor, int, int]] = []
    used = le  # number of nodes written so far, per individual

    for depth in range(last_level + 1):
        width = hi - lo
        active = _level_active(ls, le, lo, hi, dev)
        leaf_prob = float(depth2leaf_probs[depth]) if depth < n_probs else 1.0

        # ---- decide every node of this level (masked to `active`) ------
        # kernel: `rand >= leafProbs[depth]` -> function node
        r_leaf = torch.rand((pop_size, width), dtype=torch.float32, device=dev)
        is_func = r_leaf >= leaf_prob

        r_func = torch.rand((pop_size, width), dtype=torch.float32, device=dev)
        # kernel: `k = 1 + max{i : roulette_funcs[i] <= r}` (0 if none)
        func_id = torch.searchsorted(roulette_funcs, r_func, right=True).clamp_max(
            Func.END - 1
        )
        func_type = torch.where(
            func_id <= Func.IF,
            torch.full_like(func_id, NType.TFUNC),
            torch.where(
                func_id <= Func.GE,
                torch.full_like(func_id, NType.BFUNC),
                torch.full_like(func_id, NType.UFUNC),
            ),
        )
        func_value = func_id.to(torch.float32)

        if output_len > 1:
            # kernel: `rand() <= outProb` on function nodes only
            r_out = torch.rand((pop_size, width), dtype=torch.float32, device=dev)
            is_out = is_func & (r_out <= out_prob)
            out_index = torch.randint(
                0, output_len, (pop_size, width), dtype=torch.int64, device=dev
            )
            packed = (
                out_index.to(torch.int32) << _OUT_INDEX_SHIFT
            ) | func_id.to(torch.int32)
            func_value = torch.where(is_out, packed.view(torch.float32), func_value)
            func_type = torch.where(
                is_out, func_type + NType.OUT_NODE, func_type
            )

        # kernel: `rand() <= constProb` -> constant, else variable
        r_const = torch.rand((pop_size, width), dtype=torch.float32, device=dev)
        is_const = r_const <= const_prob
        const_index = torch.randint(
            0, const_samples.numel(), (pop_size, width), dtype=torch.int64, device=dev
        )
        var_index = torch.randint(
            0, input_len, (pop_size, width), dtype=torch.int64, device=dev
        )
        leaf_value = torch.where(
            is_const, const_samples[const_index], var_index.to(torch.float32)
        )
        leaf_type = torch.where(
            is_const,
            torch.full_like(func_type, NType.CONST),
            torch.full_like(func_type, NType.VAR),
        )

        new_value = torch.where(is_func, func_value, leaf_value)
        new_type = torch.where(is_func, func_type, leaf_type)
        value_grid[:, lo:hi] = torch.where(active, new_value, value_grid[:, lo:hi])
        type_grid[:, lo:hi] = torch.where(
            active, new_type.to(torch.int16), type_grid[:, lo:hi]
        )
        levels.append((ls, le, lo, hi))
        used = le

        # ---- open the next level with the children of this level -----------
        arity = _arity_from_type(type_grid[:, lo:hi]) * active.to(torch.int64)
        children = arity.sum(dim=1)
        next_le = le + children
        if int(next_le.max()) > gp_len:
            raise RuntimeError(
                f"generate_population: the generated tree does not fit in "
                f"max_tree_len={gp_len}: after depth {depth} up to "
                f"{int(next_le.max())} nodes are needed.\n"
                "This descriptor lets trees grow deeper than max_tree_len "
                "allows. Check max_layer_cnt/max_tree_len against the maximum "
                "arity of using_funcs, and make sure depth2leaf_probs contains "
                "a 1.0 terminator at max_layer_cnt - 1 (check_tree_length does "
                "that for max_layer_cnt <= MAX_FULL_DEPTH)."
            )
        if int(children.max()) == 0:
            break  # every node of this level is a leaf -> the tree is complete

        ls, le = le, next_le
        lo, hi = int(ls.min()), int(le.max())

    # ---- bottom-up subtree sizes -------------------------------------------
    # `subtree_size[i] = 1 + sum(subtree_size of the children of i)`, children
    # of one node being consecutive cells.
    size_grid = torch.zeros((pop_size, gp_len), dtype=torch.int32, device=dev)
    for level_ls, level_le, level_lo, level_hi in reversed(levels):
        active = _level_active(level_ls, level_le, level_lo, level_hi, dev)
        base, arity = _child_base(type_grid[:, level_lo : level_hi], active, level_le)
        size = torch.ones_like(arity)
        for rank in range(MAX_ARITY):
            has_child = arity > rank
            child_col = (base + rank).clamp_(0, gp_len - 1)
            child_size = torch.gather(size_grid, 1, child_col)
            size = size + torch.where(has_child, child_size, 0)
        size_grid[:, level_lo : level_hi] = torch.where(
            active, size, size_grid[:, level_lo : level_hi]
        )

    # ---- top-down prefix (pre-order) indices -------------------------------
    # prefix_index(child rank r) = prefix_index(parent) + 1 + sizes of the
    # children with a smaller rank.
    pre_grid = torch.zeros((pop_size, gp_len), dtype=torch.int64, device=dev)
    pre_flat = pre_grid.view(-1)
    row_offset = (torch.arange(pop_size, dtype=torch.int64, device=dev) * gp_len).view(
        pop_size, 1
    )
    for level_ls, level_le, level_lo, level_hi in levels:
        active = _level_active(level_ls, level_le, level_lo, level_hi, dev)
        base, arity = _child_base(type_grid[:, level_lo : level_hi], active, level_le)
        child_pre = pre_grid[:, level_lo : level_hi] + 1
        for rank in range(MAX_ARITY):
            has_child = arity > rank
            child_col = (base + rank).clamp_(0, gp_len - 1)
            flat_index = (child_col + row_offset)[has_child]
            if flat_index.numel() > 0:
                pre_flat.index_put_((flat_index,), child_pre[has_child])
            child_size = torch.gather(size_grid, 1, child_col)
            child_pre = child_pre + torch.where(has_child, child_size, 0)

    # ---- scatter the live cells into the prefix encoding -------------------
    valid = torch.arange(gp_len, dtype=torch.int64, device=dev).unsqueeze(
        0
    ) < used.unsqueeze(1)
    dst = (pre_grid + row_offset)[valid]

    batch_node_value = torch.zeros((pop_size, gp_len), dtype=torch.float32, device=dev)
    batch_node_type = torch.zeros((pop_size, gp_len), dtype=torch.int16, device=dev)
    batch_subtree_size = torch.zeros((pop_size, gp_len), dtype=torch.int16, device=dev)
    batch_node_value.view(-1).index_put_((dst,), value_grid[valid])
    batch_node_type.view(-1).index_put_((dst,), type_grid[valid])
    batch_subtree_size.view(-1).index_put_((dst,), size_grid.to(torch.int16)[valid])

    return batch_node_value, batch_node_type, batch_subtree_size
