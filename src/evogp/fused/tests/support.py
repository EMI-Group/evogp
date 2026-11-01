"""Shared builders and assertions of the ``evogp.fused`` pytest net.

The cases in :mod:`evogp.fused.tests.test_fused_engine` all work on the route's
INPUT: the legacy population triple ``(batch_node_value float32,
batch_node_type int16, batch_subtree_size int16)``, each ``(pop_size,
max_tree_len)`` in prefix (pre-order) order, on a torch device.  They used to
build it through the ``Forest`` container (deleted with the ``tree/`` stack), so
the builders and the two call-proof helpers live here instead of being
copy-pasted into every case.

Nothing here imports the route or the oracle: the triple builders delegate to
:mod:`evogp.fused.cases` (the battery's own DSL, which the shipped ``check``
stage uses too, so the two cannot drift), and the oracle reaches a case through
the ``oracle`` fixture of :mod:`conftest` -- an import by PATH, skipped (never
failed) where ``test/oracle/`` is absent.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

from evogp.fused import engine as fused_engine
from evogp.fused.cases import (
    chain,
    full_ternary,
    function_id,
    node_type_of,
    out_node_value,
)

#: The device every case runs on (the route's own device; the suite is
#: device-gated by ``tests/conftest.py``).
DEVICE = torch.device("cuda")

#: ``(node_value, node_type, subtree_size)`` -- the route's positional triple.
Triple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def triple(node_value, node_type, subtree_size) -> Triple:
    """Name the three channel tensors of a population (a documentation shortcut)."""
    return (node_value, node_type, subtree_size)


def population(
    specs: Sequence[Sequence[Any]],
    *,
    pad_to: int,
    output_len: int = 1,
) -> Triple:
    """A device population triple from nested specs (every row padded to ``pad_to``).

    A spec using the ``("out", function_id, index, kids)`` idiom is an
    ``OUT_NODE``-tagged node, which is only defined for ``output_len >= 2``; the
    assertion here is the DSL's own contract (the engine's refusal is a different,
    later rule).
    """
    rows = [encode(spec, output_len) for spec in specs]
    node_value = torch.zeros((len(specs), int(pad_to)), dtype=torch.float32)
    node_type = torch.zeros((len(specs), int(pad_to)), dtype=torch.int16)
    subtree_size = torch.zeros((len(specs), int(pad_to)), dtype=torch.int16)
    for row, (values, types, sizes) in enumerate(rows):
        if len(values) > int(pad_to):
            raise AssertionError(
                f"spec {row} has {len(values)} nodes > pad_to={int(pad_to)}"
            )
        node_value[row, : len(values)] = torch.tensor(values, dtype=torch.float32)
        node_type[row, : len(values)] = torch.tensor(types, dtype=torch.int16)
        subtree_size[row, : len(values)] = torch.tensor(sizes, dtype=torch.int16)
    return (
        node_value.to(DEVICE),
        node_type.to(DEVICE),
        subtree_size.to(DEVICE),
    )


def encode(spec: Sequence[Any], output_len: int = 1):
    """``("var",i) | ("const",v) | ("fn",fid,kids) | (fid, kids) | ("out",fid,idx,kids)``.

    The bare ``(fid, kids)`` spelling is the one :mod:`evogp.fused.cases` emits
    (``chain`` / ``full_ternary`` / the adversarial rows), so a case can hand a
    shared builder straight to :func:`population` without re-wrapping it.

    The ``("out", ...)`` idiom is the multi-output one: the node's value becomes
    the packed ``struct OutNodeValue { function, outIndex }`` bit pattern and its
    type carries ``OUT_NODE`` on top of the arity kind -- exactly what the
    generator produces for a tree built at ``output_len >= 2``.
    """
    from evogp._tables import OUT_NODE

    values: list = []
    types: list = []
    sizes: list = []

    def walk(node: Sequence[Any]) -> None:
        tag = node[0]
        index = len(values)
        values.append(0.0)
        types.append(0)
        sizes.append(0)
        kids: Sequence[Any] = ()
        if isinstance(tag, int):
            values[index] = float(tag)
            types[index] = node_type_of(int(tag))
            kids = node[1]
        elif tag == "var":
            values[index], types[index] = float(node[1]), 0
        elif tag == "const":
            values[index], types[index] = float(node[1]), 1
        elif tag == "fn":
            values[index] = float(node[1])
            types[index] = node_type_of(int(node[1]))
            kids = node[2]
        elif tag == "out":
            if int(output_len) < 2:
                raise AssertionError("an OUT_NODE tag needs output_len >= 2")
            values[index] = out_node_value(int(node[1]), int(node[2]))
            types[index] = node_type_of(int(node[1])) | int(OUT_NODE)
            kids = node[3]
        else:
            raise AssertionError(f"unknown tree-spec tag {tag!r}")
        for kid in kids:
            walk(kid)
        sizes[index] = len(values) - index

    walk(spec)
    return values, types, sizes


def tagged(function_id: int, out_index: int, children: Sequence[Any]) -> Any:
    """The ``("out", ...)`` spec of one ``OUT_NODE``-tagged node."""
    return ("out", function_id, out_index, tuple(children))


def data(rows: int, columns: int, seed: int = 7) -> torch.Tensor:
    """Deterministic positive ``(rows, columns)`` device block (LOG/SQRT/DIV safe)."""
    values = torch.rand(rows, columns, generator=torch.Generator().manual_seed(int(seed)))
    return (values + 0.25).to(DEVICE)


def rows_of(rows: Triple) -> Triple:
    """The host side of a triple (the oracle's input: three CPU tables)."""
    node_value, node_type, subtree_size = rows
    return (
        node_value.detach().cpu().float(),
        node_type.detach().cpu(),
        subtree_size.detach().cpu(),
    )


def oracle_value(oracle, rows: Triple, x, *, per_row: bool = False, output_len: int = 1):
    """The independent plain-torch answer for the same tensors (value paths)."""
    host, host_x = rows_of(rows), x.detach().cpu().float()
    emit = oracle.forward_from_rows if per_row else oracle.batch_forward_from_rows
    return emit(*host, host_x, out_len=int(output_len))


def oracle_loss(oracle, rows: Triple, inputs, labels, *, use_mse: bool = True,
                output_len: int = 1):
    """The independent plain-torch SR loss for the same tensors."""
    host = rows_of(rows)
    return oracle.sr_fitness_from_rows(
        *host,
        inputs.detach().cpu().float(),
        labels.detach().cpu().float(),
        use_mse=bool(use_mse),
        out_len=int(output_len),
    )


# --- assertions: NaN/inf PLACEMENT always exact, then bitwise or within a bound -- #
def assert_same(value, expected, context: str, *, rtol: Optional[float] = None,
                atol: float = 0.0) -> None:
    """Compare against the oracle; ``rtol=None`` demands bit-identical finite values.

    A NaN *payload* is deliberately not part of the claim (the loose surface may
    write a different payload for the same NaN), so non-finite entries are pinned by
    the placement check and only finite ones are compared.

    ``rtol=None`` is the bitwise claim; ``rtol`` set compares with
    ``torch.isclose(..., rtol=rtol, atol=atol)``.  ``atol`` exists for the
    multi-output value path, whose ``(pop, rows, out_len)`` accumulator is added to
    in float32 in the kernel's own order (see ``tests/CONTEXT.md``): a pure
    relative bound would fail on a value that is itself near zero.
    """
    got = value.detach().to(torch.float32).cpu()
    want = expected.detach().to(torch.float32).cpu()
    assert tuple(got.shape) == tuple(want.shape), (
        f"{context}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    )
    for label, mask, other in (
        ("NaN", torch.isnan(got), torch.isnan(want)),
        ("inf", torch.isinf(got), torch.isinf(want)),
    ):
        assert torch.equal(mask, other), (
            f"{context}: {label} placement differs ({int(mask.sum())} vs {int(other.sum())})"
        )
    both_inf = torch.isinf(got) & torch.isinf(want)
    if bool(both_inf.any()):
        assert torch.equal(torch.sign(got[both_inf]), torch.sign(want[both_inf])), (
            f"{context}: the sign of an infinite entry differs"
        )
    finite = torch.isfinite(got) & torch.isfinite(want)
    if not bool(finite.any()):
        return
    a, b = got[finite].contiguous(), want[finite].contiguous()
    if rtol is None:
        bad, claim = a.view(torch.int32) != b.view(torch.int32), "bit-identical"
    else:
        bad = ~torch.isclose(a, b, rtol=float(rtol), atol=float(atol))
        claim = f"within rtol={float(rtol):g}" + (f" atol={float(atol):g}" if atol else "")
    if bool(bad.any()):
        first = int(bad.nonzero()[0])
        rel = float((a[first] - b[first]).abs() / b[first].abs().clamp_min(1e-30))
        raise AssertionError(
            f"{context}: {int(bad.sum())} of {a.numel()} finite values are not {claim}; "
            f"first got={float(a[first])!r} want={float(b[first])!r} "
            f"(abs={float((a[first] - b[first]).abs()):.3e}, rel={rel:.3e})"
        )


# --- call-proof helpers (a Route name is not evidence, the counters are) ---------- #
def deltas(before, after) -> Tuple[int, int]:
    """``(declines, launches)`` between two :func:`fused_cache_stats` snapshots."""
    return (
        int(after["fused_declines"] - before["fused_declines"]),
        int(after["fused_kernel_launches"] - before["fused_kernel_launches"]),
    )


def served(call, context: str):
    """Run ``call()`` and prove the Triton kernel served it (never a fallback).

    ``fused_calls`` counts ATTEMPTS and ``fused_kernel_launches`` only real
    launches, so the counters are the evidence that a number came out of the kernel.
    """
    before = fused_engine.fused_cache_stats()
    got = call()
    declines, launches = deltas(before, fused_engine.fused_cache_stats())
    assert declines == 0, (
        f"{context}: the fused route DECLINED (+{declines}), so a fallback produced "
        "these numbers -- check EVOGP_FUSED_MAX_SCRATCH_MB / EVOGP_FUSED_BLOCK_ROWS "
        "and that the population and data are device tensors"
    )
    assert launches > 0, (
        f"{context}: no Triton launch was counted -- the call never reached the kernel"
    )
    return got


def declined(call, context: str):
    """Run ``call()`` and prove it was DECLINED (exactly one decline, no launch)."""
    before = fused_engine.fused_cache_stats()
    got = call()
    declines, launches = deltas(before, fused_engine.fused_cache_stats())
    assert (declines, launches) == (1, 0), (
        f"{context}: expected exactly one decline and no launch, counted "
        f"declines={declines} launches={launches}"
    )
    return got


__all__ = [
    "DEVICE",
    "Triple",
    "assert_same",
    "chain",
    "data",
    "declined",
    "deltas",
    "encode",
    "full_ternary",
    "function_id",
    "oracle_loss",
    "oracle_value",
    "population",
    "rows_of",
    "served",
    "tagged",
    "triple",
]
