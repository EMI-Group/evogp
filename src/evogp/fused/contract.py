"""The fused route's DECLINE and CONTRACT matrix (``--stage declines``).

This module is the second half of the battery in :mod:`evogp.fused.check`: the
half that proves the route DOWNGRADES instead of returning a wrong number.  It is
imported by ``check.py`` and run through ``python -m evogp.fused.check --stage
declines``; it is not a ``__main__`` module of its own.

WHAT IT CHECKS
--------------
1. **The decline matrix.**  Every reason this route can refuse a call
   (``backend='numpy'``, an unsupported ``output_len``, a zero-sized population or
   row axis, a host ``x``, a memory envelope that cannot fit, mis-sized labels, a
   mis-sized per-row block) raises ``FusedUnsupported`` or ``ValueError`` as
   documented, together with the POSITIVE controls: a legal call IS served at
   ``pop == 1``, with the ``iree`` backend name, and at ``output_len 2`` (whose
   counters are checked too).  A single-output request over an ``OUT_NODE``-tagged
   population raises ``evogp._multi_output.MultiOutputError`` (naming row and node)
   and is NOT a decline: nothing was launched and nothing was counted.
2. **The knob surface.**  Each ``EVOGP_FUSED_*`` resolver refuses a bad spelling
   with a message naming the variable, and the three geometry defaults match
   ``evogp.fused.kernels``.
3. **The hardened contracts** the kernel/facade rounds earned, one helper each
   (:data:`CONTRACT_CHECKS`): the ``subtree_size[0]`` clamp (B1), the masked
   empty-stack pop (S8), the floored geometry knob and the never-raising stats
   snapshot (S3/S7), the byte-bounded scratch table (S1), the emitted operator table
   against ``evogp._tables.FUNCS`` (S5), the envelope's REAL bytes (S2),
   non-contiguous inputs / labels / population tables (B2), ``out_device``, and the
   decline contract itself (B3: named, counted, launch-free, deterministic).
   A helper that raises is reported as ONE failed line, not a dead matrix.
4. **The counter surface.**  ``fused_cache_stats()`` is all ``fused_*`` keys and
   ``clear_fused_cache()`` is the ONE reset entry that zeroes them.

HOW TO RUN
----------
    TRITON_CACHE_DIR=$TMPDIR/fused/tc \
    PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \
    /tmp/rocvenv/bin/python -m evogp.fused.check --stage declines

Exit 0 iff every check passed; the tail line reports ``DECLINES: <passed>/<total>``.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

from . import engine as fused_engine
from . import kernels as fused_kernels
from .cases import (
    MAX_OUT_LEN,
    full_ternary,
    function_id,
    out_node_tagged_row,
    pack_rows,
)


# --------------------------------------------------------------------------- #
# stage 2b: the decline matrix and the knob/stat contracts (cheap, no tuning)
# --------------------------------------------------------------------------- #
# Each helper appends ``(name, ok, detail)`` rows to the stage's ``checks`` list and
# is RUN WRAPPED (see ``stage_declines``), so a crash reports as one FAIL line
# instead of killing the matrix.  They reuse the battery's own machinery
# (``pack_rows`` / ``full_ternary``) on purpose: a second way of building a
# population would be a second thing to get wrong.
# --------------------------------------------------------------------------- #
def _expect(tag: str, fn, kind, needle: Optional[str] = None) -> Tuple[str, bool, str]:
    """One "this call must raise ``kind``" expectation, as a ``checks`` row."""
    try:
        fn()
    except kind as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = needle is None or needle in str(exc)
        if not ok:
            detail = f"{detail} (does not name {needle!r})"
        return (tag, ok, detail)
    except Exception as exc:  # noqa: BLE001 - the wrong type is a failure
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    return (tag, False, "did NOT raise")


def _dev_x(rows: int, input_len: int):
    """A deterministic ``(rows, input_len)`` device block (same values every call)."""
    import torch

    torch.manual_seed(4321)
    return torch.rand(int(rows), int(input_len), device="cuda")


def _fused_value(
    rows_specs: Sequence[Sequence[Any]],
    *,
    pad_to: int,
    input_len: int,
    data_rows: int,
    size0: Optional[int] = None,
    size_row: int = 0,
):
    """``evaluate_fused`` on a hand-built population, optionally with a forced size.

    ``size0`` overrides ``batch_subtree_size[size_row, 0]`` AFTER packing, which is
    the only way to build the malformed rows the kernel's clamp has to survive
    (``pack_rows`` always writes the honest subtree sizes).
    """
    node_value, node_type, subtree_size = pack_rows(rows_specs, pad_to=pad_to)
    if size0 is not None:
        subtree_size = subtree_size.clone()
        subtree_size[int(size_row), 0] = int(size0)
    return fused_engine.evaluate_fused(
        node_value,
        node_type,
        subtree_size,
        int(input_len),
        1,
        _dev_x(data_rows, input_len),
        per_row=False,
    )


def _bitwise_row(tag: str, fn_got, fn_want, *, note: str = "") -> Tuple[str, bool, str]:
    """Compare two call results BITWISE (the contract of a layout/out_device leg)."""
    import torch

    try:
        got = fn_got()
        want = fn_want()
    except Exception as exc:  # noqa: BLE001
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    same = tuple(got.shape) == tuple(want.shape) and torch.equal(got, want)
    detail = f"bitwise={same} shape {tuple(got.shape)}"
    if not same and tuple(got.shape) == tuple(want.shape):
        detail += f" max|delta|={float((got.float() - want.float()).abs().max()):.3e}"
    return (tag, bool(same), detail + ((" " + note) if note else ""))


def _stats_row(tag: str, key: str) -> Tuple[str, bool, str]:
    """A stats snapshot must NEVER raise, whatever the ambient knob says (S7)."""
    try:
        value = fused_engine.fused_cache_stats().get(key)
    except Exception as exc:  # noqa: BLE001
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    return (tag, value == -1, f"{key}={value} (-1 = present but unusable)")


def _check_row_length_clamp(checks: List[Tuple[str, bool, str]]) -> None:
    """B1: ``subtree_size[0]`` is clamped to the row (no spill, no out-of-bounds).

    The rows below are padded to EXACTLY their node count, so the clamp
    ``min(max(subtree_size[0], 0), tree_len)`` IS the honest length and the honest
    value is the assertion: a row that walked past its own cells would read the NEXT
    row's nodes (or the memory behind the last row) and produce a different number.
    ``-5`` is the floor side of the same clamp: it floors to 0, i.e. the walk does
    nothing and the value is exactly 0.0.
    """
    import torch

    add, mul = function_id("+"), function_id("*")
    tree = full_ternary(3, 4)  # exactly 13 nodes
    other = (mul, (("var", 0), (add, (("const", 1.5), ("var", 1)))))  # 5 nodes
    honest = _fused_value([tree], pad_to=13, input_len=4, data_rows=8)
    for size0 in (30000, 32767):
        forced = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=size0)
        checks.append(
            (
                f"B1: subtree_size[0]={size0} == the honest 13-node row",
                torch.equal(forced, honest),
                f"clamped to tree_len 13; "
                f"max|delta|={float((forced - honest).abs().max()):.3e}",
            )
        )
    floored = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=-5)
    empty = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=0)
    checks.append(
        (
            "B1: subtree_size[0]=-5 floors to 0 (no crash, exactly 0.0)",
            torch.equal(floored, empty) and float(floored.abs().max()) == 0.0,
            f"max|value|={float(floored.abs().max()):.3e}, same as size 0: "
            f"{torch.equal(floored, empty)}",
        )
    )
    spilled = _fused_value([tree, other], pad_to=13, input_len=4, data_rows=8, size0=32767)
    honest_pair = _fused_value([tree, other], pad_to=13, input_len=4, data_rows=8)
    checks.append(
        (
            "B1: 32767 on a row FOLLOWED by another tree does not spill",
            torch.equal(spilled, honest_pair),
            f"max|delta|={float((spilled - honest_pair).abs().max()):.3e} "
            "(row 1 holds a different tree, so an overshoot would show)",
        )
    )


def _check_underflow_determinism(checks: List[Tuple[str, bool, str]]) -> None:
    """S8: a malformed tree popping an EMPTY stack yields a defined, stable value.

    ``MUL(VAR0, <nothing>)`` is two nodes: the walk reaches the binary root with an
    EMPTY stack.  The scratch block is shared per shape class, so the two junk
    populations below leave DIFFERENT bytes in the slot that pop would read -- one
    masked operand (``other=0.0``) or the other, the malformed result must be the
    same.  Before the mask it was the PREVIOUS call's leftover.
    """
    import torch

    add, mul = function_id("+"), function_id("*")

    def junk(value: float):
        """A right-leaning const chain: it leaves a ``2 * value``-ish sum on the stack."""
        node: Any = ("const", value)
        for _ in range(3):
            node = (add, (("const", value), node))
        return node

    def run(spec) -> Any:
        return _fused_value([spec], pad_to=8, input_len=4, data_rows=8)

    malformed = (mul, (("var", 0),))
    first_junk = run(junk(5.0))
    first = run(malformed)
    second_junk = run(junk(-7.0))
    second = run(malformed)
    checks.append(
        (
            "S8: an empty-stack pop is deterministic across junk fills",
            torch.equal(first, second),
            f"junk max {float(first_junk.max()):.3e} vs "
            f"{float(second_junk.max()):.3e} -> value "
            f"{float(first.min()):.3e} / {float(second.min()):.3e}",
        )
    )
    checks.append(
        (
            "S8: the masked (missing) operand is exactly 0.0",
            float(first.abs().max()) == 0.0 and float(second.abs().max()) == 0.0,
            f"max|value|={float(first.abs().max()):.3e} / "
            f"{float(second.abs().max()):.3e}",
        )
    )


def _check_geometry_knobs(checks: List[Tuple[str, bool, str]]) -> None:
    """S3/S7: an off-power-of-two block cap is floored, and a bad knob never raises.

    ``tl.arange`` needs a power of two, so ``EVOGP_FUSED_BLOCK_ROWS=100`` used to
    fail INSIDE Triton; the resolver and ``plan`` now floor it to 64.  An unusable
    spelling still refuses the call (``ValueError``) while a stats SNAPSHOT -- the
    one thing a driver prints unconditionally -- reports ``-1`` instead.
    """
    import torch

    node_value, node_type, subtree_size = pack_rows([("var", 0)], pad_to=4)
    x = torch.rand(8, 4, device="cuda")
    call = dict(
        batch_node_value=node_value,
        batch_node_type=node_type,
        batch_subtree_size=subtree_size,
        input_len=4,
        output_len=1,
    )
    names = (
        "EVOGP_FUSED_BLOCK_ROWS",
        "EVOGP_FUSED_NUM_WARPS",
        "EVOGP_FUSED_DISPATCH",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["EVOGP_FUSED_BLOCK_ROWS"] = "100"
        try:
            out = fused_engine.evaluate_fused(**call, x=x, per_row=False)
            served, detail = tuple(out.shape) == (1, 8, 1), f"shape {tuple(out.shape)}"
        except Exception as exc:  # noqa: BLE001
            served, detail = False, f"raised {type(exc).__name__}: {exc}"
        checks.append(("S3: BLOCK_ROWS=100 is SERVED (floored to 64)", served, detail))
        geom = fused_kernels.plan(1024, per_row=False, block_rows=100, num_warps=1)
        checks.append(
            ("S3: plan(block_rows=100) floors the block to 64", int(geom["block_rows"]) == 64,
             f"block_rows={geom['block_rows']} n_rb={geom['n_rb']}")
        )
        reported = fused_engine.fused_cache_stats()["fused_block_rows"]
        checks.append(
            ("S3: stats report the floored cap (64, not 100)", reported == 64, str(reported))
        )
        os.environ["EVOGP_FUSED_BLOCK_ROWS"] = "abc"
        checks.append(_stats_row("S7: fused_cache_stats() with BLOCK_ROWS=abc", "fused_block_rows"))
        checks.append(
            _expect(
                "S3: resolve_block_rows() still refuses 'abc'",
                fused_engine.resolve_block_rows,
                ValueError,
                "EVOGP_FUSED_BLOCK_ROWS",
            )
        )
        os.environ.pop("EVOGP_FUSED_BLOCK_ROWS", None)
        os.environ["EVOGP_FUSED_NUM_WARPS"] = "0"
        checks.append(_stats_row("S7: fused_cache_stats() with NUM_WARPS=0", "fused_num_warps"))
        checks.append(
            _expect(
                "S7: resolve_num_warps() still refuses '0'",
                fused_engine.resolve_num_warps,
                ValueError,
                "EVOGP_FUSED_NUM_WARPS",
            )
        )
        os.environ["EVOGP_FUSED_DISPATCH"] = "gpu-graph"
        checks.append(
            _expect(
                "S7: resolve_dispatch() refuses an unknown spelling",
                fused_engine.resolve_dispatch,
                ValueError,
                "EVOGP_FUSED_DISPATCH",
            )
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _check_scratch_budget(checks: List[Tuple[str, bool, str]]) -> None:
    """S1: the scratch table is bounded by BYTES, and a smaller budget evicts.

    It used to be bounded by an entry COUNT (8), i.e. up to 8x the envelope.  Two
    2 MiB shapes are allocated, then a call with a 1 MiB budget has to EVICT them
    rather than add to them.
    """
    import torch

    def serve(pop: int, tree_len: int, rows: int, **kwargs) -> None:
        node_value, node_type, subtree_size = pack_rows(
            [("var", 0)] * int(pop), pad_to=int(tree_len)
        )
        fused_engine.evaluate_fused(
            node_value,
            node_type,
            subtree_size,
            4,
            1,
            torch.rand(int(rows), 4, device="cuda"),
            per_row=False,
            **kwargs,
        )

    serve(64, 32, 256)  # 2 MiB of scratch
    serve(128, 16, 256)  # another 2 MiB
    stats = fused_engine.fused_cache_stats()
    budget = fused_engine.resolve_max_scratch_mb() * 2**20
    checks.append(
        (
            "S1: fused_scratch_bytes stays inside the resolved budget",
            stats["fused_scratch_bytes"] <= budget and stats["fused_scratch_entries"] >= 2,
            f"{stats['fused_scratch_bytes'] / 2**20:.2f} MiB in "
            f"{stats['fused_scratch_entries']} entries <= {budget / 2**20:.0f} MiB",
        )
    )
    before = fused_engine.fused_cache_stats()
    # A shape class the table has never seen (a reused one would be a HIT and prove
    # nothing) with a budget the entries above cannot fit in.
    serve(3, 5, 8, max_scratch_mb=1)
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "S1: a shrinking budget EVICTS instead of growing the table",
            after["fused_scratch_evictions"] > before["fused_scratch_evictions"]
            and after["fused_scratch_bytes"] <= 2**20,
            f"evictions +{after['fused_scratch_evictions'] - before['fused_scratch_evictions']}, "
            f"bytes {before['fused_scratch_bytes'] / 2**20:.2f} -> "
            f"{after['fused_scratch_bytes'] / 2**20:.2f} MiB",
        )
    )


def _check_operator_table(checks: List[Tuple[str, bool, str]]) -> None:
    """S5: the kernels' emitted operator table IS ``_tables.FUNCS``, split by arity.

    A function id with no branch evaluates to ``0.0`` silently, so this assertion is
    the battery's half of the guard (the import-time check in
    :mod:`evogp.fused.kernels` is the other half): removing a branch or adding an
    operator has to fail here.
    """
    from evogp._tables import FUNCS, Func

    declared = fused_kernels.emitted_function_ids()
    binary = tuple(f for f in FUNCS if Func.BF_START <= f < Func.UF_START)
    unary = tuple(f for f in FUNCS if Func.UF_START <= f < Func.END)
    checks.append(
        (
            "S5: emitted_function_ids() == _tables.FUNCS split by arity",
            tuple(declared["binary"]) == binary and tuple(declared["unary"]) == unary,
            f"binary {len(declared['binary'])}/{len(binary)} ids "
            f"{sorted(binary)[0]}..{sorted(binary)[-1]}, "
            f"unary {len(declared['unary'])}/{len(unary)} ids "
            f"{sorted(unary)[0]}..{sorted(unary)[-1]}",
        )
    )


def _check_noncontiguous(checks: List[Tuple[str, bool, str]]) -> None:
    """B2: a non-contiguous input / label / population block is read correctly.

    The kernel dereferences raw pointers derived from the SHAPE, so a column slice
    of a padded ``x`` (or a stride-2 population table) would be read with the wrong
    strides and yield plausible garbage -- no exception, no decline, a wrong answer.
    Every leg therefore compares against the same call on ``.contiguous()`` copies,
    bitwise, through the facade both stacks call.
    """
    import torch

    add = function_id("+")
    pop, input_len, tree_len, rows = 64, 4, 8, 32
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * pop, pad_to=tree_len
    )
    call = lambda x, per_row, **kwargs: fused_engine.evaluate_fused(  # noqa: E731
        node_value, node_type, subtree_size, input_len, 1, x, per_row=per_row, **kwargs
    )
    torch.manual_seed(99)
    padded = torch.rand(rows, input_len + 8, device="cuda")
    sliced = padded[:, :input_len]
    checks.append(
        _bitwise_row(
            "B2: batch_forward on a column slice == contiguous",
            lambda: call(sliced, False),
            lambda: call(sliced.contiguous(), False),
            note=f"(contiguous={sliced.is_contiguous()})",
        )
    )
    padded_row = torch.rand(pop, input_len + 8, device="cuda")
    row_slice = padded_row[:, :input_len]
    checks.append(
        _bitwise_row(
            "B2: forward(per_row) on a column slice == contiguous",
            lambda: call(row_slice, True),
            lambda: call(row_slice.contiguous(), True),
            note=f"(contiguous={row_slice.is_contiguous()})",
        )
    )
    labels = torch.rand(rows, 3, device="cuda")[:, :1]
    checks.append(
        _bitwise_row(
            "B2: SR_fitness with sliced inputs/labels == contiguous",
            lambda: fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1, sliced, labels
            ),
            lambda: fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1,
                sliced.contiguous(), labels.contiguous(),
            ),
            note=f"(labels contiguous={labels.is_contiguous()})",
        )
    )

    def strided(table):
        """The same values in a stride-2 view (the population tables themselves)."""
        wide = torch.zeros(
            (table.shape[0], 2 * table.shape[1]),
            dtype=table.dtype,
            device=table.device,
        )
        wide[:, ::2] = table
        return wide[:, ::2]

    strided_rows = (strided(node_value), strided(node_type), strided(subtree_size))
    checks.append(
        _bitwise_row(
            "B2: a stride-2 population equals its contiguous copy",
            lambda: fused_engine.evaluate_fused(*strided_rows, input_len, 1, sliced,
                                                per_row=False),
            lambda: fused_engine.evaluate_fused(node_value, node_type, subtree_size,
                                                input_len, 1, sliced, per_row=False),
            note=f"(population tables non-contiguous={not strided_rows[2].is_contiguous()})",
        )
    )
    checks.append(
        _bitwise_row(
            "B2: a stride-2 population serves the LOSS path too",
            lambda: fused_engine.sr_fitness_fused(*strided_rows, input_len, 1,
                                                  sliced, labels),
            lambda: fused_engine.sr_fitness_fused(node_value, node_type, subtree_size,
                                                  input_len, 1, sliced, labels),
            note="(same values, different strides)",
        )
    )


def _check_out_device(checks: List[Tuple[str, bool, str]]) -> None:
    """``out_device``: a host request returns the SAME numbers; a same-device one is a no-op."""
    import torch

    add = function_id("+")
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * 4, pad_to=8
    )
    rows = (node_value, node_type, subtree_size)
    x = _dev_x(16, 4)
    labels = _dev_x(16, 1)
    device = fused_engine.evaluate_fused(*rows, 4, 1, x, per_row=False)
    host = fused_engine.evaluate_fused(*rows, 4, 1, x, per_row=False, out_device="cpu")
    checks.append(
        (
            "out_device='cpu': evaluate_fused values are unchanged",
            host.device.type == "cpu" and torch.equal(host, device.cpu()),
            f"device {host.device} bitwise={torch.equal(host, device.cpu())}",
        )
    )
    same = fused_engine.evaluate_fused(
        *rows, 4, 1, x, per_row=False, out_device=device.device
    )
    checks.append(
        (
            "out_device == the computation device: a no-op",
            same.device == device.device and torch.equal(same, device),
            f"device {same.device} bitwise={torch.equal(same, device)}",
        )
    )
    loss = fused_engine.sr_fitness_fused(*rows, 4, 1, x, labels)
    loss_host = fused_engine.sr_fitness_fused(*rows, 4, 1, x, labels, out_device="cpu")
    close = bool(
        torch.allclose(loss_host, loss.cpu(), rtol=LOSS_RTOL, atol=0.0)
        if not torch.equal(loss_host, loss.cpu())
        else True
    )
    checks.append(
        (
            "out_device='cpu': sr_fitness_fused values are unchanged",
            loss_host.device.type == "cpu" and close,
            f"device {loss_host.device} "
            f"bitwise={torch.equal(loss_host, loss.cpu())}",
        )
    )


def _check_envelope_bytes(checks: List[Tuple[str, bool, str]]) -> None:
    """S2: the envelope charges the bytes a launch ACTUALLY allocates.

    ``p2048-L128-i32-N1024`` at ``out_len=8`` is the widest served multi-output
    shape: the OLD ``pop * tree_len * rows * 4`` formula reads 1 024 MiB and fits a
    1 050 MiB budget, while the scratch (1 024 MiB, rounded to 4 blocks of 256 lanes)
    plus the ``(pop, rows, out_len)`` value buffer (64 MiB) is 1 088 MiB and must
    decline.
    """
    pop, tree_len, rows, out_len, budget_mb = 2048, 128, 1024, 8, 1050
    geom = fused_kernels.plan(
        rows,
        per_row=False,
        block_rows=fused_engine.resolve_block_rows(),
        num_warps=fused_engine.resolve_num_warps(),
    )
    scratch = pop * int(geom["n_rb"]) * tree_len * int(geom["block_rows"]) * 4
    values = pop * rows * out_len * 4
    old = pop * tree_len * rows * 4
    checks.append(
        (
            "S2: the shape fits the PRE-FIX formula but not the real bytes",
            old <= budget_mb * 2**20 < scratch + values,
            f"old {old / 2**20:.0f} MiB <= {budget_mb} MiB < scratch "
            f"{scratch / 2**20:.0f} + values {values / 2**20:.0f} MiB",
        )
    )
    checks.append(
        _expect(
            "S2: p2048-L128 out_len=8 declines at 1050 MiB",
            lambda: fused_engine._check_envelope(
                pop, tree_len, rows, budget_mb, out_len=out_len
            ),
            fused_engine.FusedUnsupported,
            "value buffer",
        )
    )
    budget = fused_engine._check_envelope(pop, tree_len, rows, 8192)
    checks.append(
        (
            "S2: p2048-L128-i32-N1024-d7 returns the budget at 8192",
            budget == 8192,
            f"returned {budget!r}",
        )
    )


def _check_decline_contract(checks: List[Tuple[str, bool, str]]) -> None:
    """B3: a decline is a DECLINE -- named, counted, launch-free and deterministic.

    The caller owns the fallback now (the route no longer resolves one), so what the
    facade owes is exactly this: a :class:`FusedUnsupported` whose message names the
    reason, one ``fused_declines`` tick, no ``fused_kernel_launches`` tick, and the
    SAME message on a repeat call (a decline is a function of the shape, not of the
    scratch table).
    """
    import torch

    add = function_id("+")
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * 32, pad_to=64
    )
    rows = (node_value, node_type, subtree_size)
    x = _dev_x(1024, 4)
    messages = []
    before = fused_engine.fused_cache_stats()
    for _ in range(2):
        try:
            fused_engine.evaluate_fused(*rows, 4, 1, x, per_row=False, max_scratch_mb=1)
        except fused_engine.FusedUnsupported as exc:
            messages.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            messages.append(f"raised {type(exc).__name__}: {exc}")
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "B3: a declined call raises FusedUnsupported and names the envelope",
            len(messages) == 2
            and all("EVOGP_FUSED_MAX_SCRATCH_MB" in message for message in messages),
            (messages[0][:88] + "...") if messages else "did NOT raise",
        )
    )
    checks.append(
        (
            "B3: a declined call is counted as DECLINE and launches nothing",
            after["fused_declines"] - before["fused_declines"] == 2
            and after["fused_kernel_launches"] == before["fused_kernel_launches"],
            f"declines +{after['fused_declines'] - before['fused_declines']} "
            f"launches +{after['fused_kernel_launches'] - before['fused_kernel_launches']}",
        )
    )
    checks.append(
        (
            "B3: the decline message is deterministic (same shape, same words)",
            len(messages) == 2 and messages[0] == messages[1],
            "identical on the repeat call" if len(set(messages)) == 1 else "the two differ",
        )
    )
    # The non-raising twin must agree with the raising one, or a caller that DECIDES
    # (the gp stack asks before it hands a shape over) would be told the wrong thing.
    checks.append(
        (
            "B3: envelope_decline() predicts the same decline",
            bool(fused_engine.envelope_decline(32, 64, 1024, max_scratch_mb=1))
            and fused_engine.envelope_decline(32, 64, 1024, max_scratch_mb=8192) is None,
            "raising and non-raising verdicts agree on both sides of the budget",
        )
    )


#: The hardened-contract helpers ``stage_declines`` runs (wrapped, one FAIL each).
CONTRACT_CHECKS = (
    _check_row_length_clamp,
    _check_underflow_determinism,
    _check_geometry_knobs,
    _check_operator_table,
    _check_scratch_budget,
    _check_envelope_bytes,
    _check_noncontiguous,
    _check_out_device,
    _check_decline_contract,
)


def stage_declines(args: argparse.Namespace) -> int:
    """``FusedUnsupported`` on every unsupported call, and the stats/knob contracts.

    The route's whole safety story is "decline, never a wrong number", so every
    decline reason is exercised here (one case per reason), together with the
    POSITIVE controls (a legal call served at ``pop == 1``, with the ``iree`` backend
    name and at ``output_len 2``, whose counters are checked too) and the invariants
    the rest of the package relies on: ``fused_cache_stats()`` is all ``fused_*``
    keys and ``clear_fused_cache()`` is the ONE reset entry that zeroes them.

    The multi-output matrix is explicitly TWO-SIDED: ``output_len 2`` is SERVED (it
    used to be the decline case) while ``MAX_OUT_LEN + 1`` declines with a reason
    naming ``MAX_OUT_LEN`` and ``output_len 0`` is a ``ValueError`` (an unusable
    argument, not a decline).  A single-output request over an ``OUT_NODE``-tagged
    population is refused host-side before any launch, through the shared
    ``evogp._multi_output`` rule.

    The second half of the stage is the set of REGRESSION CONTRACTS the hardened
    kernel/facade earned (``CONTRACT_CHECKS``): the ``subtree_size[0]`` clamp (B1),
    the masked empty-stack pop (S8), the floored geometry knob and the never-raising
    stats snapshot (S3/S7), the byte-bounded scratch table (S1), the emitted operator
    table (S5), the envelope's real bytes (S2), non-contiguous inputs / labels /
    population tables (B2), ``out_device``, and the decline contract itself (B3).
    """
    import torch

    from evogp._multi_output import MultiOutputError

    checks: List[Tuple[str, bool, str]] = []

    def expect(name: str, fn, kind, needle: Optional[str] = None) -> None:
        checks.append(_expect(name, fn, kind, needle))

    node_value, node_type, subtree_size = pack_rows([("var", 0)], pad_to=4)
    x = torch.rand(8, 4, device="cuda")
    labels = torch.rand(8, 1, device="cuda")
    call = dict(
        batch_node_value=node_value,
        batch_node_type=node_type,
        batch_subtree_size=subtree_size,
        input_len=4,
        output_len=1,
    )
    expect(
        "decline: backend=numpy",
        lambda: fused_engine.evaluate_fused(**call, x=x, per_row=False, backend="numpy"),
        fused_engine.FusedUnsupported,
    )
    # The multi-output protocol is SERVED (out_len 2..MAX_OUT_LEN): the shape and
    # the counters, not a decline, are what this matrix pins now.
    before = fused_engine.fused_cache_stats()
    try:
        multi = fused_engine.evaluate_fused(**{**call, "output_len": 2}, x=x, per_row=False)
        shape_ok = tuple(multi.shape) == (1, 8, 2) and multi.dtype == torch.float32
        checks.append(
            (
                "served: output_len=2 (multi-output)",
                shape_ok,
                f"shape {tuple(multi.shape)} dtype {multi.dtype}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            ("served: output_len=2 (multi-output)", False, f"raised {type(exc).__name__}: {exc}")
        )
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "served: output_len=2 counted as a CALL, not a decline",
            after["fused_calls"] == before["fused_calls"] + 1
            and after["fused_declines"] == before["fused_declines"],
            f"calls +{after['fused_calls'] - before['fused_calls']} "
            f"declines +{after['fused_declines'] - before['fused_declines']}",
        )
    )
    expect(
        f"decline: output_len=MAX_OUT_LEN+1 ({MAX_OUT_LEN + 1})",
        lambda: fused_engine.evaluate_fused(
            **{**call, "output_len": MAX_OUT_LEN + 1}, x=x, per_row=False
        ),
        fused_engine.FusedUnsupported,
        needle="MAX_OUT_LEN",
    )
    expect(
        "raise: output_len=0",
        lambda: fused_engine.evaluate_fused(**{**call, "output_len": 0}, x=x, per_row=False),
        ValueError,
        needle="output_len",
    )
    # The single-output protocol refuses the multi-output (OUT_NODE) tag BEFORE any
    # launch: the kernel's type dispatch would otherwise fold a tagged type into its
    # TERNARY branch and return a plausible wrong number.
    tagged_rows = pack_rows([out_node_tagged_row()], pad_to=8)
    before_tag = fused_engine.fused_cache_stats()
    for tag, refused in (
        (
            "raise: single-output value path over an OUT_NODE-tagged population",
            lambda: fused_engine.evaluate_fused(*tagged_rows, 4, 1, x, per_row=False),
        ),
        (
            "raise: single-output loss path over an OUT_NODE-tagged population",
            lambda: fused_engine.sr_fitness_fused(*tagged_rows, 4, 1, x, labels),
        ),
    ):
        expect(tag, refused, MultiOutputError, needle="OUT_NODE")
    after_tag = fused_engine.fused_cache_stats()
    checks.append(
        (
            "raise: the OUT_NODE refusal is not a DECLINE and launches nothing",
            after_tag["fused_declines"] == before_tag["fused_declines"]
            and after_tag["fused_kernel_launches"] == before_tag["fused_kernel_launches"],
            f"declines +{after_tag['fused_declines'] - before_tag['fused_declines']} "
            f"launches +{after_tag['fused_kernel_launches'] - before_tag['fused_kernel_launches']}",
        )
    )
    expect(
        "decline: pop_size=0",
        lambda: fused_engine.evaluate_fused(
            **{**call, "batch_node_value": node_value[:0],
               "batch_node_type": node_type[:0], "batch_subtree_size": subtree_size[:0]},
            x=x,
            per_row=False,
        ),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: rows=0",
        lambda: fused_engine.evaluate_fused(**call, x=x[:0], per_row=False),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: x is a numpy array (never staged host memory)",
        lambda: fused_engine.evaluate_fused(**call, x=x.cpu().numpy(), per_row=False),
        fused_engine.FusedUnsupported,
    )
    # The envelope is a budget over the bytes a launch actually allocates (scratch
    # + value buffer + loss chart), so the check is landed with a legal but tiny
    # budget on the ``p256-L64-i32-N1024`` shape: it must name the components.
    expect(
        "decline: scratch envelope (1 MiB on p256-L64-N1024)",
        lambda: fused_engine._check_envelope(256, 64, 1024, 1),
        fused_engine.FusedUnsupported,
        needle="value buffer",
    )
    # The one-sided control: the SAME shape inside the budget RETURNS the resolved
    # budget in MiB (the entry points pass it down as the scratch table's budget).
    served_budget = fused_engine._check_envelope(256, 64, 1024, 8192)
    checks.append(
        (
            "served: same shape inside an 8192 MiB envelope",
            served_budget == 8192,
            f"returned {served_budget!r} MiB",
        )
    )
    expect(
        "raise: labels rows != inputs rows",
        lambda: fused_engine.sr_fitness_fused(**call, inputs=x, labels=labels[:4]),
        ValueError,
    )

    # POSITIVE controls: the same population IS served at pop 1 and with the iree
    # backend name (the fused route has no evoxc backend dependency).
    for tag, kwargs in (
        ("served: pop_size=1", {}),
        ("served: backend=iree", {"backend": "iree"}),
    ):
        try:
            out = fused_engine.evaluate_fused(**call, x=x, per_row=False, **kwargs)
            ok = tuple(out.shape) == (1, 8, 1) and out.dtype == torch.float32
            checks.append((tag, ok, f"shape {tuple(out.shape)} dtype {out.dtype}"))
        except Exception as exc:  # noqa: BLE001
            checks.append((tag, False, f"raised {type(exc).__name__}: {exc}"))

    # per_row=True with one mis-sized row block is a ValueError, not a decline.
    expect(
        "raise: per_row x rows != pop",
        lambda: fused_engine.evaluate_fused(**call, x=x, per_row=True),
        ValueError,
    )

    # knob readers (the geometry knobs and the launch path are the route's own)
    for name, reader, env, bad in (
        ("knob: BLOCK_ROWS", fused_engine.resolve_block_rows, "EVOGP_FUSED_BLOCK_ROWS", "abc"),
        ("knob: NUM_WARPS", fused_engine.resolve_num_warps, "EVOGP_FUSED_NUM_WARPS", "0"),
        (
            "knob: MAX_SCRATCH_MB",
            fused_engine.resolve_max_scratch_mb,
            "EVOGP_FUSED_MAX_SCRATCH_MB",
            "-1",
        ),
        ("knob: DISPATCH", fused_engine.resolve_dispatch, "EVOGP_FUSED_DISPATCH", "gpu-graph"),
    ):
        old = os.environ.get(env)
        os.environ[env] = bad
        try:
            expect(f"{name} rejects {bad!r}", lambda r=reader: r(), ValueError)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old
    for tag, ok, detail in (
        (
            "knob: resolve_block_rows default",
            fused_engine.resolve_block_rows() == fused_kernels.DEFAULT_BLOCK_ROWS,
            str(fused_engine.resolve_block_rows()),
        ),
        (
            "knob: resolve_num_warps default",
            fused_engine.resolve_num_warps() == fused_kernels.DEFAULT_NUM_WARPS,
            str(fused_engine.resolve_num_warps()),
        ),
        (
            "knob: resolve_dispatch default",
            fused_engine.resolve_dispatch() == "bound",
            fused_engine.resolve_dispatch(),
        ),
    ):
        checks.append((tag, ok, detail))

    # the hardened kernel/facade contracts (B1, B2, B3, S1, S2, S3, S5, S7, S8).
    # Wrapped: a helper that raises is ONE failed line, not a dead matrix.
    for helper in CONTRACT_CHECKS:
        try:
            helper(checks)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                (
                    f"{helper.__name__} did not run",
                    False,
                    f"raised {type(exc).__name__}: {exc}",
                )
            )

    # counters: own namespace, and one reset entry that really zeroes them
    stats = fused_engine.fused_cache_stats()
    checks.append(
        (
            "stats: every key is fused_*",
            all(key.startswith("fused_") for key in stats),
            ", ".join(sorted(stats)),
        )
    )
    fused_engine.clear_fused_cache()
    cleared = fused_engine.fused_cache_stats()
    checks.append(
        (
            "clear_fused_cache() zeroes its own counters and tables",
            cleared["fused_calls"] == 0
            and cleared["fused_declines"] == 0
            and cleared["fused_xt_entries"] == 0
            and cleared["fused_kernel_launches"] == 0,
            f"calls={cleared['fused_calls']} declines={cleared['fused_declines']} "
            f"xt_entries={cleared['fused_xt_entries']} launches={cleared['fused_kernel_launches']}",
        )
    )

    print("=" * 78)
    print("DECLINES & CONTRACTS: fused route")
    print("=" * 78)
    passed = 0
    for name, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<42} {detail}")
        passed += 1 if ok else 0
    print("-" * 78)
    print(f"DECLINES: {passed}/{len(checks)} checks passed")
    print("-" * 78)
    return 0 if passed == len(checks) else 1


