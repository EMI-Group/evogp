"""The ONE multi-output / ``OUT_NODE`` contract rule shared by the engines.

A node type carries the tag ``NType.OUT_NODE`` (bit 7) when the node writes its
own value into an output slot of a MULTI-OUTPUT tree, and the packed node value
then holds ``(out_index << 16) | function_id`` instead of a plain function id.
The protocol is only defined for ``output_len >= 2``: under ``output_len == 1``
the same bit is not a tag at all, and the legacy kernels did not mask it -- a
tagged type fell through their ``else`` branch and was evaluated as a TERNARY
``if`` node with its value reinterpreted as a variable index (see
``tests/reference_eval.py`` and ``ENGINE_SPEC.md`` §2.2/§4.4).  A population
built for ``output_len = 2`` and evaluated with ``output_len = 1`` therefore
produced a *plausible wrong number* rather than an error, which is the one
failure mode no caller can debug.

So every engine refuses it, on the host, BEFORE any decode/compile/launch, and
refuses it through this module -- one rule, one message, no per-engine copy:

=========================== ================================================== =========================
site                        granularity                                        error
=========================== ================================================== =========================
``evoxc_engine._decode_row`` one row (``types[:size_subtree[0]]``)             :class:`MultiOutputError`
``dense_engine.prepare_dense`` the population table + ``subtree_size[:, 0]``    ``DenseUnsupported``
``fused_engine._prepare``   the population tensors (device, memoized)          :class:`MultiOutputError`
=========================== ================================================== =========================

:class:`MultiOutputError` derives from BOTH ``ValueError`` (the tree engines'
contract: an undecodable population is a ``ValueError`` naming the row and the
node, like every other malformed-tree error) and ``NotImplementedError`` (the
functional stack ``evogp.gp`` does not implement the multi-output protocol at
all, and its sibling guards -- ``gp.encoding``, ``gp.evaluator``,
``gp.fused_eval`` -- raise ``NotImplementedError``), so a caller catching
either spelling keeps working.  ``evogp.gp`` mirrors this rule in
``gp/graph_eval.py`` instead of importing it: importing anything below
``evogp.tree`` executes that package's ``__init__``, which imports torch, and
the gp stack is torch-free by contract (``gp/tests/test_gp_stack_purity.py``).

Nothing here changes a numeric path: a population that passes this rule is
evaluated exactly as before, and the rule raises only for input that no engine
could interpret.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .utils import NType

__all__ = [
    "MAX_OUT_LEN",
    "MultiOutputError",
    "check_multi_output",
    "find_tagged_node",
    "out_node_message",
]

#: The widest multi-output tree the FUSED route serves.  The multi-output Triton
#: kernel accumulates into a ``(pop, rows, out_len)`` prediction buffer, so the
#: width is a tile/register budget rather than a protocol limit; a wider call
#: declines naming this value.  Re-exported as ``fused_engine.MAX_OUT_LEN`` (the
#: specialized and dense routes have no such ceiling).
MAX_OUT_LEN = 8

#: ``NType.OUT_NODE`` as a plain int (one spelling for every engine).
OUT_NODE = int(NType.OUT_NODE)


class MultiOutputError(ValueError, NotImplementedError):
    """A single-output request met a multi-output (``OUT_NODE``-tagged) node.

    Derives from both ``ValueError`` (an undecodable population, the tree
    engines' contract) and ``NotImplementedError`` (the protocol the functional
    stack does not implement), so both families of caller catch it.  The message
    names the row and the node, like every other decoder error.
    """


def out_node_message(row: int, node: int, raw_type: Any) -> str:
    """The refusal message of one ``OUT_NODE``-tagged node (see the module docstring).

    Args:
        row: the population row the node lives in.
        node: the node index inside that row (prefix order).
        raw_type: the offending ``node_type`` entry, as stored.

    Returns:
        The one message every engine raises (the wording the specialized decoder
        has always used, so existing error-parity checks keep matching).
    """
    return (
        f"row {int(row)}: node {int(node)}: node_type={int(raw_type)} carries the "
        "OUT_NODE flag but output_len == 1 -- tagged nodes are only defined for "
        "multi-output trees (the old kernel corrupted its operand stack here; see "
        "the module docstring)"
    )


def find_tagged_node(
    node_type: Any,
    row_lengths: Any = None,
    *,
    row: Optional[int] = None,
) -> Optional[Tuple[int, int, int]]:
    """The first ``OUT_NODE``-tagged live node of a host type table.

    Args:
        node_type: a 2-D ``[pop_size, max_nodes]`` integer table (a population) or
            a 1-D integer row.  Padding cells behind a tree may hold garbage, so
            the population form should be paired with ``row_lengths``.
        row_lengths: optional per-row live node counts (``subtree_size[:, 0]``, the
            same ``N`` the kernels walk); a cell at or past its row's length is
            ignored.  ``None`` treats every cell as live.
        row: the row index to report for the 1-D form (defaults to ``0``); ignored
            for the 2-D form, which knows its own row indices.

    Returns:
        ``(row, node, raw_type)`` of the first tagged live node in prefix order,
        or ``None`` when the table carries no tag.

    Raises:
        TypeError: ``node_type`` is not an integer array (a float table is a
            caller bug, not a tagged population).
        ValueError: ``row_lengths`` does not match the table's row axis, or
            ``node_type`` is neither 1-D nor 2-D.
    """
    raw = np.asarray(node_type)
    if raw.dtype.kind not in "iu":
        raise TypeError(
            "node_type must be an integer table (int16/int32), got dtype "
            f"{raw.dtype}"
        )
    tagged = (raw & OUT_NODE) != 0
    if tagged.ndim == 2:
        if row_lengths is not None:
            lengths = np.asarray(row_lengths).reshape(-1)
            if lengths.shape[0] != tagged.shape[0]:
                raise ValueError(
                    f"row_lengths has {lengths.shape[0]} entries for {tagged.shape[0]} "
                    "population rows"
                )
            tagged = tagged & (np.arange(tagged.shape[1])[None, :] < lengths[:, None])
        hits = np.argwhere(tagged)
        if hits.shape[0] == 0:
            return None
        found_row, found_node = int(hits[0][0]), int(hits[0][1])
        return found_row, found_node, int(raw[found_row, found_node])
    if tagged.ndim == 1:
        hits = np.flatnonzero(tagged)
        if hits.size == 0:
            return None
        found_node = int(hits[0])
        return (0 if row is None else int(row)), found_node, int(raw[found_node])
    raise ValueError(
        f"node_type must be a 1-D row or a 2-D [pop_size, max_nodes] table, got "
        f"shape {raw.shape}"
    )


def check_multi_output(
    node_type: Any,
    output_len: int,
    *,
    row_lengths: Any = None,
    row: Optional[int] = None,
    error: Any = None,
) -> None:
    """Validate the ``(node types, output_len)`` multi-output contract.

    ``output_len == 1`` is the single-output protocol: NO node may carry
    ``NType.OUT_NODE``, and the first live tagged node raises.  ``output_len >= 2``
    is the multi-output protocol, where the tag is the node's output index and
    nothing is refused (the tag's own validity is the decoder's job).

    Args:
        node_type: see :func:`find_tagged_node` -- the population table (2-D) or
            one row's live types (1-D).
        output_len: the output width the CALLER asked for, from the descriptor /
            ``Forest.output_len``.
        row_lengths: per-row live node counts, for the population form.
        row: the row index the 1-D form reports (defaults to ``0``).
        error: the exception class (or ``message -> Exception`` callable) the
            caller's contract raises -- ``DenseUnsupported`` for the dense route,
            which must DECLINE into the fallback chain rather than raise.  ``None``
            raises :class:`MultiOutputError`.

    Raises:
        MultiOutputError: (or ``error(...)``) naming the row, the node and the raw
            type, when a single-output request sees a tagged node.
        ValueError: ``output_len < 1`` (a caller bug in every engine).
    """
    if int(output_len) < 1:
        raise ValueError(f"output_len must be >= 1, got {int(output_len)}")
    if int(output_len) != 1:
        return
    found = find_tagged_node(node_type, row_lengths, row=row)
    if found is None:
        return
    found_row, found_node, raw_type = found
    message = out_node_message(found_row, found_node, raw_type)
    raise (MultiOutputError if error is None else error)(message)
