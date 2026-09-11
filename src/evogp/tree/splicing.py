"""Pure-torch tree splicing primitives (subtree replacement).

This module is a drop-in, kernel-free replacement for the two CUDA/HIP ops of
the legacy compiled extension (``tree_mutate`` and ``tree_crossover``):
it implements the exact same splice as the legacy kernel's ``_gpTreeReplace``
(distilled in ``ENGINE_SPEC.md`` §7)
with vectorized torch ops, so it runs on any torch device (CPU as well as
CUDA/HIP) and needs no compiled extension.

It deliberately works on raw ``(pop_size, max_tree_len)`` tensors (there is no
``Forest`` import) so that ``forest.py`` can call it without creating an import
cycle:

    from .splicing import mutate, crossover

Representation (identical to the kernels)
-----------------------------------------
A tree is stored in prefix (pre-order) encoding in three parallel
``[pop_size, max_tree_len]`` tensors:

* ``node_value``    - ``float32``; variable index, constant value or function id.
* ``node_type``     - ``int16``; ``NType`` base value, possibly OR-ed with
  ``NType.OUT_NODE`` (bit 7).
* ``subtree_size``  - ``int16``; ``subtree_size[i]`` is the number of nodes of the
  subtree rooted at node ``i``, so ``subtree_size[row, 0]`` is the real tree
  length of that row.

Only the first ``subtree_size[row, 0]`` columns are meaningful; the rest of the
row is padding (uninitialised garbage on the CUDA generation path).

Splice semantics (reproduced from the legacy ``_gpTreeReplace``, ``ENGINE_SPEC.md`` §7)
-------------------------------------------------------------------------------------
Replacing the subtree rooted at node ``pos`` of a target row (real length
``old_size``, subtree size ``old_sub = subtree_size[pos]``) by a donor subtree of
size ``new_sub`` starting at donor column ``donor_pos``:

1. ``size_diff = new_sub - old_sub`` and ``old_offset = pos + old_sub``.
2. Nodes ``[0, pos)`` of the target are kept verbatim.
3. ``subtree_size`` of *every strict ancestor* of ``pos`` is increased by
   ``size_diff`` (the target node itself gets the donor's own subtree sizes); the
   node at ``pos`` is therefore the only node whose subtree size is not
   ``old + size_diff``.
4. The donor columns ``[donor_pos, donor_pos + new_sub)`` are copied verbatim to
   ``[pos, pos + new_sub)``.
5. The target tail ``[old_offset, old_size)`` is shifted by ``size_diff``, i.e. it
   is copied to ``[old_offset + size_diff, old_size + size_diff)``.
6. The real length of the result is ``old_size + size_diff``; a row for which
   ``old_size + size_diff > max_tree_len`` is left untouched.

Checks and deviations from the kernels (documented on purpose)
--------------------------------------------------------------
* Index tensors accept any integer dtype (``int8/16/32/64``, ``uint8``), a
  python list/tuple, or a numpy array; the indices are converted to ``long`` and
  moved to the data device internally.  Floating point index tensors are accepted
  only when every value is integral (this keeps the legacy ``check_tensor``
  behaviour, which converts lists to ``float32``, working).
* ``left_indices`` (crossover) must be inside ``[0, pop_size)`` - the kernel reads
  those rows out of bounds, which is undefined behaviour, so a clear
  ``ValueError`` is raised instead.
* An out-of-range ``right_indices`` entry (negative or ``>= pop_size``) is *not*
  an error: the kernel checks it and copies the left parent row, and so does this
  module.
* The kernel performs no bounds check on ``left_pos``/``right_pos`` (out of range
  reads are undefined); here an out-of-range position silently makes the row fall
  back to the "bad pos" path (output row = the left/target parent row copy).
* For rows that must stay unchanged (bad position or overflow) the kernel writes
  only the ``[0, old_size)`` prefix; this module copies the *entire* row verbatim
  (all ``max_tree_len`` columns) so the result is deterministic.
* For a valid row, columns ``>= new_size`` are not written by the kernel
  (``torch::empty`` garbage); here they are set to the target row's original
  values.  Only ``[0, subtree_size[row, 0])`` is semantically meaningful.
* A degenerate donor (``subtree_size[:, 0] <= 0``) is applied through the same
  formulas as the kernel instead of writing out of bounds.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = ["mutate", "crossover"]

# integer dtypes accepted for the index tensors
_INDEX_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _normalize_tree(
    node_value: Tensor,
    node_type: Tensor,
    subtree_size: Tensor,
    name: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Validate a raw tree batch and return it as ``(f32, i16, i16)`` tensors."""
    for tensor, tname, is_value in (
        (node_value, "node_value", True),
        (node_type, "node_type", False),
        (subtree_size, "subtree_size", False),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}.{tname} must be a torch.Tensor, but got {type(tensor)}"
            )
        if tensor.dim() != 2:
            raise ValueError(
                f"{name}.{tname} must be 2-D [pop_size, max_tree_len], "
                f"but got shape {tuple(tensor.shape)}"
            )
        if is_value:
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name}.{tname} must be a floating point tensor, "
                    f"but got dtype {tensor.dtype}"
                )
        elif tensor.dtype not in _INDEX_DTYPES:
            raise TypeError(
                f"{name}.{tname} must be an integer tensor, but got dtype {tensor.dtype}"
            )

    if not (node_value.shape == node_type.shape == subtree_size.shape):
        raise ValueError(
            f"{name}.node_value / node_type / subtree_size must have identical "
            f"shapes, but got {tuple(node_value.shape)}, {tuple(node_type.shape)}, "
            f"{tuple(subtree_size.shape)}"
        )

    return (
        node_value.to(torch.float32),
        node_type.to(torch.int16),
        subtree_size.to(torch.int16),
    )


def _as_index(index, name: str, device: torch.device) -> Tensor:
    """Convert an index argument (Tensor / list / numpy array) to a 1-D long tensor."""
    if not isinstance(index, torch.Tensor):
        index = torch.as_tensor(index)

    if index.dtype not in _INDEX_DTYPES:
        if not index.is_floating_point():
            raise TypeError(
                f"{name} must contain integers, but got dtype {index.dtype}"
            )
        if bool((index != index.trunc()).any()):
            raise TypeError(
                f"{name} must contain integers, but got non-integral values "
                f"of dtype {index.dtype}"
            )

    index = index.to(dtype=torch.long, device=device)
    if index.dim() != 1:
        raise ValueError(f"{name} must be 1-D, but got shape {tuple(index.shape)}")
    return index


def _splice(
    target_value: Tensor,
    target_type: Tensor,
    target_size: Tensor,
    target_pos: Tensor,
    donor_value: Tensor,
    donor_type: Tensor,
    donor_size: Tensor,
    donor_pos: Tensor,
    new_subtree_size: Tensor,
    extra_valid: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Vectorized ``_gpTreeReplace``: splice one donor subtree into every target row.

    All tree tensors are ``[pop_size, max_tree_len]`` (value ``float32``,
    type/size ``int16``) and ``target_pos`` / ``donor_pos`` /
    ``new_subtree_size`` / ``extra_valid`` are 1-D ``[pop_size]`` tensors
    (``long`` / ``bool``).  Rows that are invalid (position out of range, result
    longer than ``max_tree_len``, or ``extra_valid`` false) are copied verbatim.

    Returns freshly allocated ``(value, type, size)`` tensors.
    """
    max_tree_len = target_value.shape[1]
    device = target_value.device

    cols = torch.arange(max_tree_len, device=device, dtype=torch.long).unsqueeze(0)
    # out-of-range positions are clamped for the gathers and dropped by `valid`
    pos = target_pos.clamp(0, max_tree_len - 1).unsqueeze(1)
    donor_start = donor_pos.clamp(0, max_tree_len - 1).unsqueeze(1)
    new_sub = new_subtree_size.unsqueeze(1).to(torch.long)

    size_old = target_size.to(torch.long)
    old_size = size_old[:, 0:1]
    old_sub = size_old.gather(1, pos)
    size_diff = new_sub - old_sub
    new_size = old_size + size_diff

    valid = (target_pos >= 0) & (target_pos < old_size.squeeze(1))
    valid = valid & (new_size.squeeze(1) <= max_tree_len)
    if extra_valid is not None:
        valid = valid & extra_valid
    keep = valid.unsqueeze(1) & (cols < new_size)

    # index maps: donor block vs. (shifted) target tail
    in_donor = (cols >= pos) & (cols < pos + new_sub)
    donor_col = torch.clamp(donor_start + cols - pos, 0, max_tree_len - 1)
    old_col = torch.clamp(
        torch.where(cols < pos + new_sub, cols, cols - size_diff),
        0,
        max_tree_len - 1,
    )

    value = torch.where(
        in_donor, donor_value.gather(1, donor_col), target_value.gather(1, old_col)
    )
    node_type = torch.where(
        in_donor, donor_type.gather(1, donor_col), target_type.gather(1, old_col)
    )

    # node j < pos is a strict ancestor of pos iff its subtree contains pos, and
    # every strict ancestor grows by size_diff.  All other nodes keep their own
    # subtree size, but the shifted tail ones are read at their old position
    # (`old_col`); the replaced node itself comes from the donor block.
    ancestors = (cols < pos) & (cols + size_old > pos)
    subtree_size = size_old.gather(1, old_col)
    subtree_size = torch.where(ancestors, subtree_size + size_diff, subtree_size)
    subtree_size = torch.where(
        in_donor, donor_size.to(torch.long).gather(1, donor_col), subtree_size
    )

    # unchanged rows: whole row = target row; valid rows: tail = target row
    value = torch.where(keep, value, target_value)
    node_type = torch.where(keep, node_type, target_type)
    subtree_size = torch.where(keep, subtree_size, size_old)

    return value, node_type.to(torch.int16), subtree_size.to(torch.int16)


def mutate(
    batch_node_value: Tensor,
    batch_node_type: Tensor,
    batch_subtree_size: Tensor,
    replace_pos: Tensor,
    donor_node_value: Tensor,
    donor_node_type: Tensor,
    donor_subtree_size: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Replace one subtree per row of a tree batch by a row of a donor batch.

    Row ``i``: the subtree rooted at ``replace_pos[i]`` of target row ``i`` is
    replaced by the whole tree of donor row ``i`` (donor columns
    ``[0, donor_subtree_size[i, 0])``).  This is what the legacy extension's
    ``tree_mutate`` op did.

    Args:
        batch_node_value: target ``[pop_size, max_tree_len]`` ``float32`` values.
        batch_node_type: target ``[pop_size, max_tree_len]`` ``int16`` node types.
        batch_subtree_size: target ``[pop_size, max_tree_len]`` ``int16`` subtree sizes.
        replace_pos: ``[pop_size]`` position of the subtree to replace in every
            target row.  An integer tensor (``int32``/``int64``/...), a list or a
            numpy array; converted to ``long`` internally.
        donor_node_value: donor ``[pop_size, max_tree_len]`` ``float32`` values.
        donor_node_type: donor ``[pop_size, max_tree_len]`` ``int16`` node types.
        donor_subtree_size: donor ``[pop_size, max_tree_len]`` ``int16`` subtree sizes.

    Returns:
        A fresh ``(node_value, node_type, subtree_size)`` tuple with the shape of
        the target batch (``float32`` / ``int16`` / ``int16``), on the same device
        as the inputs.  Only ``[0, subtree_size[row, 0])`` of each row is
        semantically meaningful (see the module docstring for the padding rule).

    Raises:
        TypeError: if a tensor has an unsupported dtype or is not a tensor.
        ValueError: on wrong shapes, on an empty batch, or when ``replace_pos``
            does not have shape ``(pop_size,)``.

    Notes:
        A row whose ``replace_pos`` is negative or beyond its real tree length
        (``subtree_size[row, 0]``), or whose result would exceed ``max_tree_len``
        columns, is returned as a verbatim copy of the target row - silently, as
        in the kernel.  The donor is copied with the donor's own subtree sizes,
        so the ``OUT_NODE`` flag and the node payloads are preserved bit for bit.
    """
    value, node_type, subtree_size = _normalize_tree(
        batch_node_value, batch_node_type, batch_subtree_size, "target"
    )
    donor_value, donor_type, donor_size = _normalize_tree(
        donor_node_value, donor_node_type, donor_subtree_size, "donor"
    )

    pop_size = value.shape[0]
    if pop_size == 0:
        raise ValueError("pop_size must be greater than 0, but got 0")
    if donor_value.shape != value.shape:
        raise ValueError(
            f"donor tensors must have the target shape {tuple(value.shape)}, "
            f"but got {tuple(donor_value.shape)}"
        )

    pos = _as_index(replace_pos, "replace_pos", value.device)
    if pos.shape != (pop_size,):
        raise ValueError(
            f"replace_pos must have shape ({pop_size},), but got {tuple(pos.shape)}"
        )

    new_subtree_size = donor_size[:, 0].to(torch.long)
    donor_pos = torch.zeros_like(pos)
    return _splice(
        value,
        node_type,
        subtree_size,
        pos,
        donor_value,
        donor_type,
        donor_size,
        donor_pos,
        new_subtree_size,
    )


def crossover(
    batch_node_value: Tensor,
    batch_node_type: Tensor,
    batch_subtree_size: Tensor,
    left_indices: Tensor,
    right_indices: Tensor,
    left_pos: Tensor,
    right_pos: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Replace a left-parent subtree by a right-parent subtree, row by row.

    Output row ``i`` is a copy of target row ``left_indices[i]`` in which the
    subtree rooted at ``left_pos[i]`` is replaced by the subtree rooted at
    ``right_pos[i]`` of donor row ``right_indices[i]``.  This is what the legacy
    extension's ``tree_crossover`` op did; the output population has
    ``len(left_indices)`` rows.

    Args:
        batch_node_value: ``[pop_size, max_tree_len]`` ``float32`` values.
        batch_node_type: ``[pop_size, max_tree_len]`` ``int16`` node types.
        batch_subtree_size: ``[pop_size, max_tree_len]`` ``int16`` subtree sizes.
        left_indices: ``[pop_size_out]`` rows that provide the left parents.
        right_indices: ``[pop_size_out]`` rows that provide the donor subtrees.
        left_pos: ``[pop_size_out]`` splice position inside the left parent.
        right_pos: ``[pop_size_out]`` donor subtree root inside the right parent.

    Returns:
        A fresh ``(node_value, node_type, subtree_size)`` tuple of shape
        ``(len(left_indices), max_tree_len)`` (``float32`` / ``int16`` /
        ``int16``), on the same device as the inputs.

    Raises:
        TypeError: if a tensor has an unsupported dtype or is not a tensor.
        ValueError: on wrong shapes, when ``left_indices`` is empty, or when a
            ``left_indices`` entry is outside ``[0, pop_size)``.

    Notes:
        A row is left as a verbatim copy of its left parent if the right parent
        index is out of range (negative or ``>= pop_size``, exactly as checked by
        the kernel), if ``left_pos``/``right_pos`` is outside the real tree length
        of the corresponding parent, or if the result would exceed
        ``max_tree_len`` columns.
    """
    value, node_type, subtree_size = _normalize_tree(
        batch_node_value, batch_node_type, batch_subtree_size, "parents"
    )
    pop_size, max_tree_len = value.shape
    device = value.device

    left_idx = _as_index(left_indices, "left_indices", device)
    right_idx = _as_index(right_indices, "right_indices", device)
    lpos = _as_index(left_pos, "left_pos", device)
    rpos = _as_index(right_pos, "right_pos", device)

    if left_idx.shape[0] == 0:
        raise ValueError("left_indices must not be empty")
    for tensor, name in (
        (right_idx, "right_indices"),
        (lpos, "left_pos"),
        (rpos, "right_pos"),
    ):
        if tensor.shape != left_idx.shape:
            raise ValueError(
                f"{name} must have shape {tuple(left_idx.shape)} "
                f"(the same as left_indices), but got {tuple(tensor.shape)}"
            )

    if bool((left_idx < 0).any()) or bool((left_idx >= pop_size).any()):
        raise ValueError(
            f"left_indices must be in range [0, {pop_size}), but got "
            f"{left_idx.tolist()}"
        )

    # out-of-range right rows are handled per row by the kernel: copy the left parent
    right_valid = (right_idx >= 0) & (right_idx < pop_size)
    right_idx_safe = torch.clamp(right_idx, 0, pop_size - 1)

    target_value = torch.index_select(value, 0, left_idx)
    target_type = torch.index_select(node_type, 0, left_idx)
    target_size = torch.index_select(subtree_size, 0, left_idx)

    donor_value = torch.index_select(value, 0, right_idx_safe)
    donor_type = torch.index_select(node_type, 0, right_idx_safe)
    donor_size = torch.index_select(subtree_size, 0, right_idx_safe)

    pos_safe = torch.clamp(rpos, 0, max_tree_len - 1).unsqueeze(1)
    new_subtree_size = (
        donor_size.gather(1, pos_safe).squeeze(1).to(torch.long)
    )

    extra_valid = (
        right_valid
        & (lpos >= 0)
        & (lpos < target_size[:, 0].to(torch.long))
        & (rpos >= 0)
        & (rpos < donor_size[:, 0].to(torch.long))
    )

    return _splice(
        target_value,
        target_type,
        target_size,
        lpos,
        donor_value,
        donor_type,
        donor_size,
        rpos,
        new_subtree_size,
        extra_valid,
    )
