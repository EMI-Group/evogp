#!/usr/bin/env python3
"""Re-execute the tutorial notebook and check its STORED outputs against a fresh run.

`tutorial/gp_intro.ipynb` ships the outputs of a real execution, so its "stored
outputs re-executed against the current library" claim needs a reproducible check.
This script is that check and the whole toolchain: standard library only (no
jupyter, nbconvert or nbformat -- the reference venv has none of them).

It loads the notebook JSON with `json`, executes the `code` cells in order in one
shared `__main__` namespace, captures stdout per cell, turns a trailing expression
into a Jupyter-style `execute_result` (eval + `repr`), and diffs the fresh output
line by line against what the notebook stores: `stream` text plus the `text/plain`
of `execute_result`/`display_data`. Wall-clock fragments are normalized away before
the diff (`--ignore-regex`), so timing jitter is not a difference. Exit code 0 means
every code cell matched; a DIFF or an exception exits non-zero.

`--write` re-stores the fresh outputs instead (same JSON: `nbformat`/`nbformat_minor`
and every markdown cell preserved, `execution_count` set) -- refresh-only and
byte-stable, since re-serializing an unchanged notebook reproduces it exactly.

Run from the repository root, with a `PYTHONPATH` that reaches `src/` (the notebook
imports `evogp.gp`); the device cell wants the iree packages and a rocm/cuda device
and otherwise falls back to the numpy host loop:

    PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \
      TRITON_CACHE_DIR=/tmp/triton_cache_nb \
      /tmp/rocvenv/bin/python tutorial/run_notebook.py

Requirements: the project venv (`/tmp/rocvenv/bin/python`), numpy, `evoxc` and that
`PYTHONPATH`. Runtime: ~15 s on the reference ROCm box with a warm Triton cache (a
CPU-only box declines the device cell and finishes in ~4 s). The `nanobind: leaked
...` lines on stderr at interpreter shutdown are harmless.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
from itertools import zip_longest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "tutorial" / "gp_intro.ipynb"
# The wall-clock fragments the notebook prints: "in 1.05s (35.0 ms/generation)",
# "in 11.84s (first-call graph compile included)".
DEFAULT_IGNORE = r"\bin [0-9.]+ ?s\b|[0-9.]+ ?ms/generation"
TIME_TOKEN = "<time>"


def joined(payload) -> str:
    """Join an nbformat multi-line string (a list of chunks, or a plain string)."""
    return "".join(map(str, payload)) if isinstance(payload, list) else (payload or "")


def stored_lines(cell: dict) -> list[str]:
    """The output lines a code cell stores, flattened for a line-by-line diff."""
    lines: list[str] = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            lines.extend(joined(output.get("text", "")).splitlines())
        elif kind in ("execute_result", "display_data"):
            lines.extend(joined(output.get("data", {}).get("text/plain")).splitlines())
        elif kind == "error":
            lines.append(f"{output.get('ename')}: {output.get('evalue')}")
    return lines


def execute_cell(source: str, filename: str, namespace: dict) -> tuple[str, list[str]]:
    """Run one cell; return its captured stdout plus any Jupyter-style display lines."""
    module = ast.parse(source)
    # Jupyter evaluates a trailing expression and displays its repr, so the last
    # statement is compiled separately from the cell body.
    trailing = module.body[-1].value \
        if module.body and isinstance(module.body[-1], ast.Expr) else None
    head = ast.Module(body=module.body[:-1], type_ignores=[]) if trailing else source
    buffer, value = io.StringIO(), None
    with contextlib.redirect_stdout(buffer):
        exec(compile(head, filename, "exec"), namespace)
        if trailing:
            value = eval(compile(ast.Expression(trailing), filename, "eval"), namespace)
    return buffer.getvalue(), [] if value is None else repr(value).splitlines()


def diff_cell(stored: list[str], fresh: list[str], pattern: re.Pattern | None) -> list[tuple]:
    """(stored, fresh) line pairs that differ, after normalizing wall-clock fragments."""
    normalize = (lambda lines: [pattern.sub(TIME_TOKEN, line) for line in lines]) \
        if pattern else (lambda lines: lines)
    pairs = zip_longest(normalize(stored), normalize(fresh), fillvalue="<no line>")
    return [(left, right) for left, right in pairs if left != right]


def fresh_outputs(stdout: str, displays: list[str], count: int) -> list[dict]:
    """stdout and the trailing expression as nbformat outputs, the way Jupyter stores them."""
    outputs = [{"name": "stdout", "output_type": "stream",
                "text": stdout.splitlines(keepends=True)}] if stdout else []
    if displays:
        outputs.append({"output_type": "execute_result", "execution_count": count,
                        "data": {"text/plain": [line + "\n" for line in displays]},
                        "metadata": {}})
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-execute a notebook's code cells and diff against its stored outputs.")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK,
                        help=f"notebook to run (default: {DEFAULT_NOTEBOOK.relative_to(REPO_ROOT)})")
    parser.add_argument("--ignore-regex", default=DEFAULT_IGNORE,
                        help=f"wall-clock fragments to normalize away, '' to compare verbatim "
                             f"(default: {DEFAULT_IGNORE!r})")
    parser.add_argument("--write", action="store_true",
                        help="store the fresh outputs into the notebook instead of only checking")
    args = parser.parse_args(argv)

    path = args.notebook if args.notebook.is_absolute() else Path.cwd() / args.notebook
    notebook = json.loads(path.read_text(encoding="utf-8"))
    pattern = re.compile(args.ignore_regex) if args.ignore_regex else None

    namespace: dict = {"__name__": "__main__"}
    total = differing = 0
    failed: list[str] = []
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            total += 1
            stdout, displays = execute_cell("".join(cell["source"]), f"<cell {index}>", namespace)
            if args.write:
                cell["outputs"] = fresh_outputs(stdout, displays, total)
                cell["execution_count"] = total
                print(f"WROTE  cell {index}: {len(stdout.splitlines()) + len(displays)} output lines")
                continue
            differences = diff_cell(stored_lines(cell), stdout.splitlines() + displays, pattern)
            differing += len(differences)
            print(f"{'DIFF ' if differences else 'MATCH'}  cell {index}: "
                  f"{len(stdout.splitlines()) + len(displays)} output lines"
                  f"{f', {len(differences)} differing' if differences else ''}")
            for stored_line, fresh_line in differences:
                print(f"    - stored: {stored_line!r}\n    + fresh:  {fresh_line!r}")
            if differences:
                failed.append(f"cell {index}")
    except Exception as error:  # noqa: BLE001 - a cell that raises is a failed run
        print(f"ERROR  {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    if args.write:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {total}/{total} code cells into {path}")
        return 0

    print(f"\n{total - len(failed)}/{total} code cells MATCH, {differing} lines differing")
    if failed:
        print(f"differing cells: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
