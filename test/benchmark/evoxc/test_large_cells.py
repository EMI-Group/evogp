"""Torch-free checks of the large-scale column (``--cells large``).

This directory is normally not collected by pytest (no CI collects it), but the project
gate runs ``pytest test src/evogp/tests -q``, which walks into it -- so this module is
the ONE automated check of the column's shape, and it stays import-light on purpose:
``common`` / ``campaign`` only, no ``torch``, no ``evoxc``, no ``evogp``, no device.

What is pinned here (each of it a claim ``common.large_cells`` makes in its docstring):

* the ten ids are exactly the documented grid, in the documented order;
* every id round-trips through ``common.parse_cell`` into an equal cell dict, is unique
  and is spelled ``p<pop>-L<tree_len>-i<input_len>-N<rows>[-d<depth>]``;
* the ``-d<k>`` suffix appears exactly on the two raised-depth cells and nowhere else;
* the memory rule (``pop * tree_len * rows * 4 <= CELL_TENSOR_CEILING_BYTES``) holds
  for every cell, and the documented worst cell really is the worst one;
* the column covers each of its four documented axes;
* ``campaign.select_cells`` maps ``"large"`` to this column and still parses a single
  cell id, i.e. the token wiring is testable without a device.
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # pytest's prepend mode normally does this; be explicit
    sys.path.insert(0, _HERE)

import campaign  # noqa: E402  (sibling harness module, same directory)
import common  # noqa: E402

#: The documented id spelling: a ``-d<k>`` suffix only above the default depth cap of 5.
CELL_ID_RE = re.compile(r"^p\d+-L\d+-i\d+-N\d+(-d\d+)?$")

#: The grid of record, as ``(pop, tree_len, input_len, rows, max_layer_cnt)`` in the
#: documented order -- four axis groups: population scale, row scale, node-count scale
#: at large pop, input width.
EXPECTED_GRID = (
    (2048, 64, 32, 1024, 5),
    (4096, 64, 32, 1024, 5),
    (8192, 64, 32, 1024, 5),
    (256, 64, 32, 4096, 5),
    (256, 64, 32, 16384, 5),
    (1000, 64, 32, 4096, 5),
    (1000, 64, 32, 16384, 5),
    (2048, 128, 32, 1024, 7),
    (2048, 256, 32, 1024, 8),
    (4096, 64, 64, 1024, 5),
)


def _shape(cell):
    return (
        cell["pop"],
        cell["tree_len"],
        cell["input_len"],
        cell["rows"],
        cell["max_layer_cnt"],
    )


def test_grid_is_exactly_the_documented_one():
    """The ten cells, in the documented order (the column is a record, not a generator)."""
    cells = common.large_cells()
    assert [_shape(cell) for cell in cells] == list(EXPECTED_GRID)
    assert len(cells) == 10


def test_ids_round_trip_through_parse_cell():
    for cell in common.large_cells():
        assert common.parse_cell(cell["id"]) == cell


def test_ids_are_unique_and_spelled_like_a_cell():
    ids = [cell["id"] for cell in common.large_cells()]
    assert len(set(ids)) == len(ids)
    for cell_id in ids:
        assert CELL_ID_RE.match(cell_id), cell_id


def test_depth_suffix_only_on_the_raised_caps():
    for cell in common.large_cells():
        has_suffix = bool(re.search(r"-d\d+$", cell["id"]))
        assert has_suffix == (cell["max_layer_cnt"] != 5), cell["id"]
    suffixed = {cell["id"] for cell in common.large_cells() if cell["max_layer_cnt"] != 5}
    assert suffixed == {"p2048-L128-i32-N1024-d7", "p2048-L256-i32-N1024-d8"}


def test_memory_rule_holds_for_every_cell():
    """``pop * tree_len * rows * 4`` stays inside the documented per-cell ceiling."""
    ceiling = common.CELL_TENSOR_CEILING_BYTES
    sizes = {}
    for cell in common.large_cells():
        nbytes = common.cell_tensor_bytes(cell)
        # the formula itself is the rule: input_len does not enter it
        assert nbytes == cell["pop"] * cell["tree_len"] * cell["rows"] * 4
        assert nbytes <= ceiling, f"{cell['id']} needs {nbytes / 2**30:.2f} GiB"
        sizes[cell["id"]] = nbytes
    worst = max(sizes, key=sizes.get)
    # the docstring names this cell as the worst one of the column
    assert worst == "p1000-L64-i32-N16384"
    assert sizes[worst] == 4_194_304_000
    # and the ceiling keeps a real margin, not a hair's breadth
    assert sizes[worst] * 2 <= ceiling


def test_column_covers_each_documented_axis():
    cells = common.large_cells()
    assert any(cell["pop"] >= 4096 for cell in cells), "population scale"
    assert any(cell["rows"] >= 4096 for cell in cells), "row scale"
    assert {cell["max_layer_cnt"] for cell in cells} & {7, 8}, "node-count scale at d7/d8"
    assert any(cell["input_len"] == 64 for cell in cells), "input width"
    # the node-count cells must sit above the dense route's default trace cap, otherwise
    # they would price nothing that the main grid does not already price
    assert all(
        cell["tree_len"] > 64
        for cell in cells
        if cell["max_layer_cnt"] in (7, 8)
    )


def test_select_cells_maps_the_large_token():
    assert campaign.select_cells("large") == common.large_cells()
    # the existing tokens keep their meaning
    assert campaign.select_cells("long") == common.long_tree_cells()
    assert campaign.select_cells("n64") == common.n64_cells()
    assert campaign.select_cells("main") == common.main_cells()
    # a single id still parses, alone and next to a column
    single = campaign.select_cells("p256-L64-i32-N1024")
    assert single == [common.cell(256, 64, 32, 1024)]
    assert campaign.select_cells("large,p256-L64-i32-N1024") == common.large_cells() + single
    assert campaign.select_cells("") == []


#: The population-scaling column of record: ONE fixed shape (tree_len 32, input_len 8,
#: rows 1024, depth cap 4 -- the ``evogp.gp`` acceptance config) and eight populations.
POPSCALE_GRID = tuple((pop, 32, 8, 1024, 4) for pop in (64, 128, 256, 512, 1024, 2048, 4096, 8192))


def test_popscale_grid_is_exactly_the_documented_one():
    """Eight cells, doubling population, every shape field but ``pop`` held fixed."""
    cells = common.popscale_cells()
    assert [_shape(cell) for cell in cells] == list(POPSCALE_GRID)
    assert [cell["pop"] for cell in cells] == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # the column IS the population axis: no other field moves, and pop always doubles
    assert len({(c["tree_len"], c["input_len"], c["rows"], c["max_layer_cnt"]) for c in cells}) == 1
    assert all(b == 2 * a for a, b in zip([c["pop"] for c in cells], [c["pop"] for c in cells][1:]))


def test_popscale_ids_round_trip_and_carry_the_depth_suffix():
    cells = common.popscale_cells()
    ids = [cell["id"] for cell in cells]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "p64-L32-i8-N1024-d4"
    assert ids[-1] == "p8192-L32-i8-N1024-d4"
    for cell in cells:
        assert common.parse_cell(cell["id"]) == cell
        assert CELL_ID_RE.match(cell["id"]), cell["id"]
        # depth cap 4 is not the default 5, so every id carries ``-d4``
        assert cell["id"].endswith("-d4")
        # the generated trees (2 ** 4 - 1 = 15 nodes) fit the row stride and dense's cap
        assert 2 ** cell["max_layer_cnt"] - 1 <= cell["tree_len"] <= 64


def test_popscale_memory_rule_holds_for_every_cell():
    ceiling = common.CELL_TENSOR_CEILING_BYTES
    sizes = {}
    for cell in common.popscale_cells():
        nbytes = common.cell_tensor_bytes(cell)
        assert nbytes == cell["pop"] * cell["tree_len"] * cell["rows"] * 4
        assert nbytes <= ceiling, f"{cell['id']} needs {nbytes / 2**30:.2f} GiB"
        sizes[cell["id"]] = nbytes
    worst = max(sizes, key=sizes.get)
    assert worst == "p8192-L32-i8-N1024-d4"
    assert sizes[worst] == 1 << 30  # exactly 1 GiB
    assert sizes[worst] * 2 <= ceiling


def test_select_cells_maps_the_popscale_token():
    assert campaign.select_cells("popscale") == common.popscale_cells()
    assert "popscale" in campaign.CELL_COLUMNS
    single = campaign.select_cells("p8192-L32-i8-N1024-d4")
    assert single == [common.cell(8192, 32, 8, 1024, 4)]
    assert campaign.select_cells("popscale,p64-L32-i8-N1024-d4") == common.popscale_cells() + [
        common.cell(64, 32, 8, 1024, 4)
    ]
