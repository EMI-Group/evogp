"""Dev battery for the FUSED (Triton) evaluation route — parity, goldens, contract, timing.

This is a ``__main__``-guarded CLI (``python -m evogp.fused.check``), not a pytest
module: it needs a CUDA/HIP device, it is the route's EXHAUSTIVE numeric and
contract battery, and it is the only place where the fused route's numbers and its
economics are both stated.  Nothing imports it, and every leg it spawns re-enters
through :func:`main`.

WHAT IT CHECKS
--------------
1. **Parity against the oracle** (``--stage parity``).  The independent plain-torch
   reference is ``<repo>/test/oracle/reference_eval.py`` (dev-only tooling, imported
   by PATH by :func:`_setup_paths`): written without the engines, and the numeric
   contract of record.  Five random populations cover the function-table families --
   the arithmetic set, the arithmetic+ternary set, the loose/protected set, a
   ``rich`` transcendental set and a ``compare_unary`` set (``>``, ``<=``, ``neg``,
   ``abs``, ``sinh``, ``cosh``, ``+``, ``*``) that is the ONLY place the six operators
   no other set draws are exercised -- and the handmade adversarial rows of
   :func:`evogp.fused.cases.adversarial_cases` cover the shapes a random generator
   does not produce: single-node leaves, a pure constant tree, a 40-deep left chain,
   a 40-deep right chain, full ternary ``if`` trees and a real tree followed by
   garbage padding.  Each population is compared on all three surfaces: the batched
   value path (``per_row=False``), the per-tree value path (``per_row=True``) and
   the SR loss (MSE and MAE).  Every ``PARITY_SETS`` entry carries its own MEASURED
   ``value_tol`` (``0.0`` = the value path must be bitwise, which is what the
   kernels' ``enable_fp_fusion=False`` buys), every adversarial case its own measured
   ``bitwise`` flag, and the loss path is ``<= 1e-5`` relative everywhere (the kernel
   reduces in float32 per row block, the oracle in one float32 ``sum``, so the two
   differ by accumulation order only).
2. **The multi-output protocol** (``--stage multi``).  Handmade ``OUT_NODE``-tagged
   populations (``cases.multi_adversarial_cases``) at ``2 <= out_len <= MAX_OUT_LEN``:
   both outputs of a deep/shallow pair, a tagged root ternary, a single tagged output
   (the others must stay exactly ``0.0``), two nodes accumulating into ONE output,
   tagged leaves (which accumulate nothing), an out-of-range ``outIndex`` (dropped),
   one tagged node per arity, a tree with no tag at all (every output ``0.0``, so the
   loss is the label norm), a packed ``MUL`` beside a packed ``ADD`` on the same
   operands, and one population evaluated at ``out_len`` 2 AND 3.  All four surfaces
   run against the oracle; the value path must stay BITWISE.
3. **Goldens** (``--stage goldens``).  The frozen ``.npz`` fixtures of
   ``<repo>/test/oracle/goldens`` (the oracle's own schema) at the documented ``1e-4``
   rtol/atol, plus ``$EVOGP_GOLDEN_DIR`` when it is set.  Multi-output fixtures are
   SKIPPED here -- this is the single-output golden gate -- so the multi-output
   protocol is verified by ``--stage multi`` against the oracle instead.  The
   measured reading is ``6/6``: the oracle's seven fixtures, with
   ``06_multi_output.npz`` skipped for being multi-output.
4. **Declines and the knob/stat contracts** (``--stage declines``).  The route's whole
   safety story is "decline, never a wrong number": every decline reason is exercised
   here, together with the POSITIVE controls (a legal call served at ``pop == 1``,
   with the ``iree`` backend name, and at ``output_len 2``) and the regression
   contracts the hardening rounds earned -- ``subtree_size[0]`` clamped to the row
   (B1), an empty-stack pop a deterministic masked ``0.0`` (S8), an off-power-of-two
   ``EVOGP_FUSED_BLOCK_ROWS`` floored and an unusable knob reported as ``-1`` by the
   stats snapshot (S3/S7), the scratch table bounded by BYTES and evicting on a
   smaller budget (S1), ``emitted_function_ids()`` exactly ``utils.FUNCS`` by arity
   (S5), the envelope charging the bytes a launch really allocates (S2), non-contiguous
   inputs / labels / population tables giving the same BITWISE numbers as their
   ``.contiguous()`` copies (B2), ``out_device`` changing placement only, and a
   declined call raising, counting one decline and launching nothing (B3).
5. **Timings** (``--stage timing``).  Six cells, warm medians (2 warmup + 10
   sync-bracketed reps) of ``evaluate_fused`` / ``sr_fitness_fused`` and of the
   per-tree value path, the cold first call (Triton import + JIT) and a stage split
   (kernel vs host glue vs the cold transpose).  One process per cell, because a warm
   median is only comparable if every process sees one shape class.  The stage
   reports **ABSOLUTE milliseconds**: the ``torch-batched`` reference arm lives in the
   ARCHIVED benchmark harness (``test/archive/benchmark/evoxc``), which this package
   deliberately does not reach into and which is not runnable from here, so there is
   no ratio column -- compare against a number you measured yourself, on the same
   box, in the same session.
6. **Tuning** (``--stage tuning``).  ``EVOGP_FUSED_BLOCK_ROWS`` in {64,128,256,512}
   x ``EVOGP_FUSED_NUM_WARPS`` in {1,2,4} on the same six cells, one process per
   (config, cell), also in absolute milliseconds.  The configs are ranked by their
   WORST per-cell ratio against the SHIPPED default (256 rows, 1 warp) -- a route
   ships one default, not six, so the minimax answer is the one that is never much
   dearer than what ships.

LAYOUT
------
``check.py`` (this module) is the CLI plus the numeric legs (``parity`` / ``multi``
/ ``goldens``); ``contract.py`` is the decline + contract matrix; ``timing.py`` is
the timing and tuning stages with their single-process children and the cell
table; ``cases.py`` holds every shared case builder, the tolerances and the timing
tool.  No leg re-implements a population builder -- a second way of building one
would be a second thing to get wrong.

HOW TO RUN
----------
From the repository root, with the torch-carrying interpreter (``TRITON_CACHE_DIR``
must be writable -- the default ``~/.triton`` is not, in the sandbox)::

    TRITON_CACHE_DIR=$TMPDIR/fused/tc \\
    PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \\
    /tmp/rocvenv/bin/python -m evogp.fused.check --stage parity
    ... --stage multi           # the multi-output (OUT_NODE) protocol
    ... --stage goldens
    ... --stage declines
    ... --stage timing          # six cells, one subprocess each
    ... --stage tuning          # 4 x 3 configs x six cells
    ... --stage all

``--stage time-one --cell <id>`` / ``--stage tune-one --cell <id> --block B
--warps W`` are the single-process legs the two drivers spawn; they print one
``RESULT {...json...}`` line and are not meant to be run by hand except when a cell
misbehaves.

MEASURED TOLERANCES (re-measured for this package)
--------------------------------------------------
* The value paths are BITWISE against the oracle for ``arithmetic``,
  ``compare_unary``, the adversarial rows and every multi-output case
  (``output_len`` 2 / 3 / 8, both value paths).
* ``ternary`` / ``loose`` / ``rich`` measured ``max_rel = 0`` as well and keep the
  ``1e-7`` margin a transcendental emitter might need.
* The LOSS path (single- and multi-output) is not bitwise -- the kernel reduces in
  float32 per row block, the oracle in one float32 sum: relative ``<= 1.2e-07`` on
  every case measured here (absolute up to 3.1e-05 on the deepest adversarial rows,
  where the values are large), i.e. inside the ``1e-5`` gate by two orders of
  magnitude.
* The oracle is skipped (with a clear message) when ``<repo>/test/oracle`` is absent
  -- only the ``goldens`` and ``declines`` stages then remain runnable.

DEVIATIONS FROM THE ORACLE THAT ARE NOT BUGS
--------------------------------------------
* The oracle raises on a VAR index outside ``[0, input_len)``; the kernel cannot (a
  device read out of bounds would be undefined), so it masks the load with
  ``other=0.0``.  Every handmade row here keeps its VAR indices in range.
* ``rows == 0`` and ``pop == 0`` are declines, not empty results -- the route refuses
  to launch a zero-sized grid.
* ``FusedUnsupported`` is a DECLINE (the facade's contract): a call it cannot serve
  reports the decline and never a number, and the CALLER owns the fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import engine as fused_engine
from . import kernels as fused_kernels
from .cases import (
    DEFAULT_PARITY_POP,
    DEFAULT_PARITY_ROWS,
    GOLDEN_ATOL,
    GOLDEN_RTOL,
    LOSS_RTOL,
    PARITY_SETS,
    VALUE_RTOL,
    adversarial_cases,
    compare,
    multi_adversarial_cases,
    pack_rows,
    workload,
)
from .contract import stage_declines
from .timing import (
    SHIPPED_BLOCK_ROWS,
    SHIPPED_NUM_WARPS,
    stage_time_one,
    stage_timing,
    stage_tune_one,
    stage_tuning,
)

# --------------------------------------------------------------------------- #
# paths (the cells, the drivers and the contract matrix live beside this module)
# --------------------------------------------------------------------------- #
#: ``<repo>/src/evogp/fused`` -> the repository root, where ``test/`` lives.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_ORACLE_DIR = os.path.join(_REPO_ROOT, "test", "oracle")
_ORACLE_FILE = os.path.join(_ORACLE_DIR, "reference_eval.py")

# --------------------------------------------------------------------------- #
# the oracle, by path
# --------------------------------------------------------------------------- #
def _setup_paths() -> None:
    """Make ``<repo>/test/oracle`` importable (the oracle is DEV tooling).

    Called from :func:`main` -- and therefore from every ``-m`` child leg the driver
    stages spawn -- so merely IMPORTING this module never mutates ``sys.path``.
    The oracle is deliberately NOT on ``PYTHONPATH``: it is shipped nowhere and
    imported by no library module, and a route that reached it from library code
    would be depending on dev tooling.
    """
    if os.path.isdir(_ORACLE_DIR) and _ORACLE_DIR not in sys.path:
        sys.path.append(_ORACLE_DIR)


def _oracle(stage: str):
    """The oracle module, or a clean :class:`SystemExit` naming the missing path."""
    try:
        import reference_eval  # type: ignore
    except Exception as exc:  # noqa: BLE001 - an environment error, never a silent pass
        raise SystemExit(
            f"--stage {stage} compares against the plain-torch oracle, which is "
            f"missing or broken: {exc!r}\n  expected {_ORACLE_FILE} (dev-only "
            "tooling, see test/oracle/CONTEXT.md)"
        )
    return reference_eval


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _json_line(payload: Dict[str, Any]) -> None:
    """One machine-readable line for the driver stages to parse."""
    print("RESULT " + json.dumps(payload, default=str), flush=True)


def _dev_block(rows: int, columns: int, seed: int):
    """A deterministic ``(rows, columns)`` device block (same values every process)."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    return torch.rand(int(rows), int(columns), generator=generator).cuda()


def _report(
    tag: str,
    stats: Dict[str, Any],
    limit: float,
    *,
    key: str,
    sink: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Print one comparison row and return whether it stayed inside ``limit``.

    ``sink`` collects the same row for a stage that wants to aggregate it (the
    ``--stage multi`` table); the parity stage passes ``None``.
    """
    ok = (stats.get("nan_mismatch", 1) == 0) and (stats.get("inf_mismatch", 1) == 0)
    if "shape_mismatch" in stats:
        ok = False
    rel = float(stats.get("max_rel", 0.0))
    ok = ok and rel <= limit
    mark = "ok " if ok else "FAIL"
    print(
        f"  [{mark}] {tag:<34} bitwise={str(stats.get('bitwise_equal')):<5} "
        f"max_abs={stats.get('max_abs', 0.0):.3e} max_rel={rel:.3e} "
        f"nan_mismatch={stats.get('nan_mismatch', 0)} "
        f"inf_mismatch={stats.get('inf_mismatch', 0)} "
        f"({key} <= {limit:g})",
        flush=True,
    )
    if sink is not None:
        sink.append(dict(stats, tag=tag, key=key, limit=float(limit), ok=bool(ok)))
    return ok


# --------------------------------------------------------------------------- #
# stage 1: parity vs the oracle
# --------------------------------------------------------------------------- #
def _oracle_rows(rows: Tuple[Any, Any, Any]):
    node_value, node_type, subtree_size = rows
    return node_value.cpu(), node_type.cpu(), subtree_size.cpu()


def _oracle_value(rows: Tuple[Any, Any, Any], x, out_len: int = 1, *, per_row: bool = False):
    oracle = _oracle("parity")
    node_value, node_type, subtree_size = _oracle_rows(rows)
    emit = oracle.forward_from_rows if per_row else oracle.batch_forward_from_rows
    return emit(node_value, node_type, subtree_size, x.cpu(), out_len=int(out_len))


def _oracle_loss(rows, x, labels, use_mse: bool, out_len: int = 1):
    oracle = _oracle("parity")
    node_value, node_type, subtree_size = _oracle_rows(rows)
    return oracle.sr_fitness_from_rows(
        node_value,
        node_type,
        subtree_size,
        x.cpu(),
        labels.cpu(),
        use_mse=bool(use_mse),
        out_len=int(out_len),
    )


def _check_population(
    tag: str,
    rows: Tuple[Any, Any, Any],
    input_len: int,
    *,
    batch_x=None,
    per_row_x=None,
    labels=None,
    value_limit: float,
    note: str = "",
    out_len: int = 1,
    sink: Optional[List[Dict[str, Any]]] = None,
    capture: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """All three surfaces of one population.  Returns ``(passed, total)``.

    ``out_len`` selects the protocol: ``1`` (the default, and every call site of the
    parity stage) is the single-output one, ``>= 2`` the ``OUT_NODE`` protocol the
    multi-output kernel implements.  ``sink`` collects the comparison rows and
    ``capture`` the raw ``(got, want)`` tensors, both for ``--stage multi``.
    """
    node_value, node_type, subtree_size = rows
    passed = total = 0
    if note:
        print(f"  {tag}: {note}", flush=True)
    if batch_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, out_len, batch_x, per_row=False
        )
        want = _oracle_value(rows, batch_x, out_len)
        if capture is not None:
            capture["batch_forward"] = (got, want)
        total += 1
        passed += _report(
            f"{tag}/batch_forward",
            compare(want, got),
            value_limit,
            key="value",
            sink=sink,
        )
        if labels is not None:
            for use_mse in (True, False):
                got_l = fused_engine.sr_fitness_fused(
                    node_value, node_type, subtree_size, input_len, out_len, batch_x, labels,
                    use_MSE=use_mse,
                )
                want_l = _oracle_loss(rows, batch_x, labels, use_mse, out_len=out_len)
                if capture is not None:
                    capture["sr_fitnessMSE" if use_mse else "sr_fitnessMAE"] = (
                        got_l,
                        want_l,
                        labels,
                    )
                total += 1
                passed += _report(
                    f"{tag}/sr_fitness{'MSE' if use_mse else 'MAE'}",
                    compare(want_l, got_l),
                    LOSS_RTOL,
                    key="loss",
                    sink=sink,
                )
    if per_row_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, out_len, per_row_x, per_row=True
        )
        want = _oracle_value(rows, per_row_x, out_len, per_row=True)
        if capture is not None:
            capture["forward(per_row)"] = (got, want)
        total += 1
        passed += _report(
            f"{tag}/forward(per_row)",
            compare(want, got),
            value_limit,
            key="value",
            sink=sink,
        )
    return passed, total


def stage_parity(args: argparse.Namespace) -> int:
    """Random op-set populations + the handmade adversarial rows."""
    import torch

    oracle_message = _ORACLE_FILE
    print("=" * 78)
    print("PARITY: fused engine vs the plain-torch oracle")
    print(f"  oracle: {oracle_message}")
    print("=" * 78)
    passed = total = 0
    for name, spec in PARITY_SETS.items():
        input_len = int(spec["input_len"])
        cell = {
            "pop": int(args.parity_pop),
            "tree_len": 32,
            "input_len": input_len,
            "rows": int(args.parity_rows),
            "max_layer_cnt": int(spec["max_layer_cnt"]),
        }
        built = workload(cell, using_funcs=list(spec["using_funcs"]))
        rows = built["rows"]
        print(f"\n[{name}] using_funcs={list(spec['using_funcs'])}")
        limit = float(spec["value_tol"])
        p, t = _check_population(
            name,
            rows,
            input_len,
            batch_x=built["x"],
            per_row_x=built["x_per_row"],
            labels=built["labels"],
            value_limit=limit,
            note=(
                f"bitwise required (measured value_tol={limit:g})"
                if limit == 0.0
                else f"value <= {limit:g} rel (measured 0 with a margin)"
            ),
        )
        passed += p
        total += t
    print("\n[adversarial]")
    for case in adversarial_cases():
        node_value, node_type, subtree_size = pack_rows(
            case["rows"],
            pad_to=int(case.get("pad_to", 0)),
            garbage_tail=bool(case.get("garbage_tail", False)),
        )
        input_len = int(case["input_len"])
        pop = int(node_value.shape[0])
        torch.manual_seed(1234)
        batch_x = torch.rand(int(args.parity_rows), input_len, device="cuda")
        labels = torch.rand(int(args.parity_rows), 1, device="cuda")
        per_row_x = torch.rand(pop, input_len, device="cuda")
        print(f"\n[{case['name']}]")
        p, t = _check_population(
            case["name"],
            (node_value, node_type, subtree_size),
            input_len,
            batch_x=batch_x,
            per_row_x=per_row_x,
            labels=labels,
            value_limit=0.0 if case["bitwise"] else VALUE_RTOL,
            note=case["note"],
        )
        passed += p
        total += t
    print("\n" + "-" * 78)
    print(f"PARITY: {passed}/{total} comparisons inside tolerance")
    print("-" * 78)
    return 0 if passed == total else 1


# --------------------------------------------------------------------------- #
# stage 1b: the multi-output (OUT_NODE) protocol
# --------------------------------------------------------------------------- #
def _check_line(tag: str, ok: bool, detail: str = "") -> int:
    """Print one structural expectation of ``--stage multi``; ``1`` when it holds."""
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:<40} {detail}", flush=True)
    return 1 if ok else 0


def _multi_case_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a case's comparison rows: the value bounds and the loss bound."""
    value = [row for row in rows if row["key"] == "value"]
    loss = [row for row in rows if row["key"] == "loss"]
    return dict(
        value_max_abs=max((float(row.get("max_abs", 0.0)) for row in value), default=0.0),
        value_max_rel=max((float(row.get("max_rel", 0.0)) for row in value), default=0.0),
        value_bitwise=bool(value) and all(bool(row.get("bitwise_equal")) for row in value),
        loss_max_rel=max((float(row.get("max_rel", 0.0)) for row in loss), default=0.0),
        loss_max_abs=max((float(row.get("max_abs", 0.0)) for row in loss), default=0.0),
        nan_mismatch=sum(int(row.get("nan_mismatch", 0)) for row in rows),
        inf_mismatch=sum(int(row.get("inf_mismatch", 0)) for row in rows),
    )


def stage_multi(args: argparse.Namespace) -> int:
    """The multi-output protocol (``out_len`` 2..``MAX_OUT_LEN``) vs the oracle.

    The route serves ``output_len`` 1..:data:`evogp.fused.MAX_OUT_LEN`: ``out_len
    == 1`` runs the single-output kernel (what ``--stage parity`` pins), ``out_len
    >= 2`` the kernel that resolves a tagged node's operator from the packed value
    bits, accumulates into ``out[outIndex]``, drops an index ``>= out_len``, ignores
    tagged leaves and leaves untagged outputs at ``0.0``.

    This stage runs ``cases.multi_adversarial_cases()`` -- the shapes a random
    generator does not produce for that protocol -- on all four surfaces: the batched
    value path, the per-tree value path and BOTH loss paths (MSE and MAE), each
    against the plain-torch oracle.  The value path must be BITWISE
    (``value_limit = 0.0``: the multi-output kernel reuses the single-output per-op
    emitters) and the loss ``<= LOSS_RTOL`` relative.

    Cases also carry structural expectations a numeric comparison alone would not
    pin: outputs no node tags must be EXACTLY ``0.0``, a population with no tag must
    lose exactly ``sum(labels**2) / rows``, and two rows that differ only in their
    packed operator must actually produce different numbers.
    """
    import torch

    print("=" * 78)
    print("MULTI: the multi-output (OUT_NODE) protocol vs the plain-torch oracle")
    print(f"  oracle: {_ORACLE_FILE}")
    print("=" * 78)
    passed = total = 0
    records: List[Dict[str, Any]] = []
    for case in multi_adversarial_cases():
        rows = pack_rows(
            case["rows"],
            pad_to=int(case.get("pad_to", 0)),
            garbage_tail=bool(case.get("garbage_tail", False)),
        )
        input_len = int(case["input_len"])
        pop = int(rows[0].shape[0])
        out_lens = tuple(int(item) for item in case.get("out_lens") or (case["out_len"],))
        for out_len in out_lens:
            tag = f"{case['name']}/out{out_len}"
            torch.manual_seed(1234)
            batch_x = torch.rand(int(args.parity_rows), input_len, device="cuda")
            labels = torch.rand(int(args.parity_rows), out_len, device="cuda")
            per_row_x = torch.rand(pop, input_len, device="cuda")
            sink: List[Dict[str, Any]] = []
            capture: Dict[str, Any] = {}
            print(f"\n[{tag}] ops={case['ops']}")
            p, t = _check_population(
                tag,
                rows,
                input_len,
                batch_x=batch_x,
                per_row_x=per_row_x,
                labels=labels,
                value_limit=float(case.get("value_limit", 0.0)),
                note=case["note"],
                out_len=out_len,
                sink=sink,
                capture=capture,
            )
            passed += p
            total += t
            batch_got = capture["batch_forward"][0]
            per_row_got = capture["forward(per_row)"][0]
            # 1. every output no node tags must be EXACTLY 0.0 in BOTH value paths.
            expect_zero = case.get("expect_zero", ())
            if isinstance(expect_zero, dict):
                expect_zero = expect_zero.get(out_len, ())
            for index in expect_zero:
                batch_max = float(batch_got[:, :, index].abs().max())
                per_row_max = float(per_row_got[:, index].abs().max())
                total += 1
                passed += _check_line(
                    f"{tag}/out{index}==0.0",
                    max(batch_max, per_row_max) == 0.0,
                    f"max|batch|={batch_max:.3e} max|per_row|={per_row_max:.3e}",
                )
            # 2. a population with no tag at all predicts 0.0 everywhere, so its MSE
            #    is exactly the label norm (an independent, oracle-free cross-check).
            if case.get("expect_loss_is_label_norm"):
                got_loss = capture["sr_fitnessMSE"][0]
                expected = float((labels.to(torch.float32) ** 2).sum()) / float(
                    labels.shape[0]
                )
                ok = bool(
                    torch.allclose(
                        got_loss.to(torch.float32),
                        torch.full_like(got_loss, expected),
                        rtol=LOSS_RTOL,
                        atol=0.0,
                    )
                )
                total += 1
                passed += _check_line(
                    f"{tag}/loss==label_norm",
                    ok,
                    f"expected {expected:.6e}, got {float(got_loss.min()):.6e}.."
                    f"{float(got_loss.max()):.6e}",
                )
            # 3. two rows that differ only in their packed operator must differ.
            if case.get("expect_rows_differ"):
                left, right = case["expect_rows_differ"]
                delta = float((batch_got[left] - batch_got[right]).abs().max())
                total += 1
                passed += _check_line(
                    f"{tag}/rows {left} vs {right} differ",
                    delta > 0.0,
                    f"max|delta|={delta:.3e}",
                )
            aggregate = _multi_case_stats(sink)
            line_ok = all(bool(row["ok"]) for row in sink)
            print(
                f"  [{'ok ' if line_ok else 'FAIL'}] CASE {tag:<38} out_len={out_len} "
                f"ops={case['ops']:<16} max_abs={aggregate['value_max_abs']:.3e} "
                f"max_rel={aggregate['value_max_rel']:.3e} "
                f"bitwise={str(aggregate['value_bitwise']):<5} "
                f"loss_max_rel={aggregate['loss_max_rel']:.3e} "
                f"nan_mismatch={aggregate['nan_mismatch']}",
                flush=True,
            )
            records.append(
                dict(
                    case=case["name"],
                    out_len=out_len,
                    ops=case["ops"],
                    note=case["note"],
                    comparisons=len(sink),
                    comparisons_ok=line_ok,
                    rows=[
                        dict(
                            tag=row["tag"],
                            key=row["key"],
                            limit=row["limit"],
                            ok=row["ok"],
                            max_abs=row.get("max_abs", 0.0),
                            max_rel=row.get("max_rel", 0.0),
                            bitwise_equal=row.get("bitwise_equal"),
                            nan_mismatch=row.get("nan_mismatch", 0),
                            inf_mismatch=row.get("inf_mismatch", 0),
                        )
                        for row in sink
                    ],
                    **aggregate,
                )
            )
    print("\n" + "-" * 78)
    print(f"MULTI: {passed}/{total} comparisons inside tolerance")
    print("-" * 78)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0 if passed == total else 1


# --------------------------------------------------------------------------- #
# stage 2: goldens
# --------------------------------------------------------------------------- #
_ENGINE_KEYS = ("node_value", "node_type", "subtree_size")


def _golden_dirs() -> List[str]:
    """The fixture directories this package reads.

    ``<repo>/test/oracle/goldens`` ALWAYS (the oracle's own frozen cases, the current
    schema) and ``$EVOGP_GOLDEN_DIR`` for a fixture set kept elsewhere.
    """
    directories = [os.path.join(_ORACLE_DIR, "goldens")]
    extra = os.environ.get("EVOGP_GOLDEN_DIR")
    if extra:
        directories.append(extra)
    return [directory for directory in directories if os.path.isdir(directory)]


def stage_goldens(args: argparse.Namespace) -> int:
    """The frozen ``.npz`` fixtures at the documented 1e-4 rtol/atol."""
    import numpy as np
    import torch

    passed = total = 0
    directories = _golden_dirs()
    print("=" * 78)
    print("GOLDENS: fused engine vs the frozen fixtures (1e-4 rtol/atol)")
    print("=" * 78)
    if not directories:
        print(
            "  [FAIL] no fixture directory found; expected "
            f"{os.path.join(_ORACLE_DIR, 'goldens')} (or $EVOGP_GOLDEN_DIR)"
        )
        return 1
    for directory in directories:
        print(f"  fixtures: {directory}")
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".npz"):
                continue
            path = os.path.join(directory, name)
            with np.load(path) as data:
                keys = set(data.files)
                if set(_ENGINE_KEYS) <= keys:
                    passed, total = _golden_engine(
                        path, data, compare, fused_engine, torch, passed, total
                    )
                else:
                    print(f"  [skip] {name}: unrecognized schema ({sorted(keys)[:6]}...)")
    print("-" * 78)
    print(f"GOLDENS: {passed}/{total} comparisons inside 1e-4")
    print("-" * 78)
    return 0 if passed == total else 1


def _as_cuda(values, dtype):
    import torch

    return torch.as_tensor(values, dtype=dtype).cuda()


def _golden_engine(path, data, compare, fused_engine, torch, passed, total):
    """``test/oracle/goldens/*.npz``: the value / SR-loss expectations."""
    name = os.path.basename(path)
    x = _as_cuda(data["x"], torch.float32)
    node_value = _as_cuda(data["node_value"], torch.float32)
    node_type = _as_cuda(data["node_type"], torch.int16)
    subtree_size = _as_cuda(data["subtree_size"], torch.int16)
    input_len = int(x.shape[1])
    multi_output = "expected" in data and int(np_shape_last(data["expected"])) > 1
    if multi_output:
        print(
            f"  [skip] {name}: multi-output fixture (this stage is the single-output "
            "golden gate; --stage multi covers the protocol)"
        )
        return passed, total
    if "expected" in data:
        want = _as_cuda(data["expected"], torch.float32)
        try:
            got = fused_engine.evaluate_fused(
                node_value, node_type, subtree_size, input_len, 1, x, per_row=False
            )
        except fused_engine.FusedUnsupported as exc:
            print(f"  [FAIL] {name}/expected: declined ({exc})")
            total += 1
            return passed, total
        total += 1
        passed += _golden_report(f"{name}/expected", compare(want, got), name)
    if "expected_fitness" in data:
        labels = _as_cuda(data["labels"], torch.float32)
        use_mse = bool(int(np_scalar(data["use_mse"]))) if "use_mse" in data else True
        want = _as_cuda(data["expected_fitness"], torch.float32)
        try:
            got = fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1, x, labels,
                use_MSE=use_mse,
            )
        except fused_engine.FusedUnsupported as exc:
            print(f"  [FAIL] {name}/expected_fitness: declined ({exc})")
            total += 1
            return passed, total
        total += 1
        passed += _golden_report(
            f"{name}/expected_fitness(mse={use_mse})", compare(want, got), name
        )
    return passed, total


def _golden_report(tag: str, stats: Dict[str, Any], fixture: str) -> int:
    """``GOLDEN_RTOL``/``GOLDEN_ATOL`` -- the frozen fixtures carry no bitwise claim."""
    ok = (
        (stats.get("nan_mismatch", 1) == 0)
        and (stats.get("inf_mismatch", 1) == 0)
        and "shape_mismatch" not in stats
        and float(stats.get("max_abs", 1e9)) <= GOLDEN_ATOL + GOLDEN_RTOL * float(
            stats.get("ref_abs_max", 0.0)
        )
    )
    rel = float(stats.get("max_rel", 0.0))
    absd = float(stats.get("max_abs", 0.0))
    print(
        f"  [{'ok ' if ok else 'FAIL'}] {tag:<44} max_abs={absd:.3e} max_rel={rel:.3e} "
        f"bitwise={stats.get('bitwise_equal')} nan_mismatch={stats.get('nan_mismatch', 0)}",
        flush=True,
    )
    return 1 if ok else 0


def np_shape_last(array) -> int:
    import numpy as np

    return int(np.asarray(array).shape[-1])


def np_scalar(array) -> float:
    import numpy as np

    return float(np.asarray(array).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evogp.fused.check",
        description="Dev battery of the fused (Triton) evaluation route.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "parity",
            "multi",
            "goldens",
            "declines",
            "timing",
            "tuning",
            "all",
            "time-one",
            "tune-one",
        ),
        help="which leg to run (default: all)",
    )
    parser.add_argument("--cell", default=None, help="cell id for the single-process legs")
    parser.add_argument("--block", type=int, default=None,
                        help="EVOGP_FUSED_BLOCK_ROWS for tune-one (default: the shipped one)")
    parser.add_argument(
        "--warps",
        type=int,
        default=None,
        help="EVOGP_FUSED_NUM_WARPS for tune-one (default: fused_kernels.DEFAULT_NUM_WARPS)",
    )
    parser.add_argument(
        "--blocks", default="64,128,256,512", help="BLOCK_ROWS grid of the tuning driver"
    )
    parser.add_argument("--warps-list", default="1,2,4", help="num_warps grid of the tuning driver")
    parser.add_argument(
        "--parity-pop", type=int, default=DEFAULT_PARITY_POP, help="pop of a parity population"
    )
    parser.add_argument(
        "--parity-rows", type=int, default=DEFAULT_PARITY_ROWS, help="data rows of a parity set"
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated cell ids to restrict the driver stages to",
    )
    parser.add_argument("--json", default=None, help="write the stage's records to this file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_paths()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.only:
        args.only = [item.strip() for item in args.only.split(",") if item.strip()]
    if args.stage in ("time-one", "tune-one") and not args.cell:
        parser.error(f"--stage {args.stage} needs --cell")
    if args.block is None:
        args.block = int(SHIPPED_BLOCK_ROWS)
    if args.warps is None:
        # ``--warps``' default IS the shipped kernel default (single-sourced).
        args.warps = int(SHIPPED_NUM_WARPS)
    status = 0
    if args.stage in ("parity", "all"):
        status |= stage_parity(args)
    if args.stage in ("multi", "all"):
        status |= stage_multi(args)
    if args.stage in ("goldens", "all"):
        status |= stage_goldens(args)
    if args.stage in ("declines", "all"):
        status |= stage_declines(args)
    if args.stage in ("timing", "all"):
        status |= stage_timing(args)
    if args.stage in ("tuning", "all"):
        status |= stage_tuning(args)
    if args.stage == "time-one":
        status |= stage_time_one(args)
    if args.stage == "tune-one":
        status |= stage_tune_one(args)
    print(f"\nevogp.fused.check: exit {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
