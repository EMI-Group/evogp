"""Public-API regression check for ``Tree.forward`` / ``Tree.SR_fitness``.

Hand-built, RNG-free specimen kept from the original ``Tree.SR_fitness``
mode-agreement repro: an 8-node prefix tree that decodes to
``(x0 - x2) * (x0 - x2)``.  Nothing in this file draws from a generator, so every
number below is deterministic and can be asserted exactly.

Checks:
  * ``Tree.forward`` reproduces the closed form of the decoded tree, both for a
    2-D ``(batch, input_len)`` batch and for a single 1-D ``(input_len,)`` row;
  * ``Tree.SR_fitness`` agrees with the loss recomputed from ``Tree.forward`` (MSE
    and MAE) and keeps the documented convention: positive loss, summed over the
    data points, never divided by ``output_len``;
  * every ``execute_mode`` string (``"torch"`` included) routes to the same engine
    path and therefore returns bit-identical numbers - the legacy ``kernel_type``
    dispatch is gone and the strings are pure API compatibility;
  * the batched wrapper (``Tree.to_forest()``) reports the same loss;
  * an unknown ``execute_mode`` is rejected.

Usage (from the repository root; a CUDA-visible GPU is required because the
public API relocates every tensor to ``"cuda"``)::

    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python test/fix_bug.py

Exit code 0 means all checks passed; a failure names the check that broke.
"""

import torch

from evogp.tree import Tree

DEVICE = "cuda"

#: Every accepted ``execute_mode`` string; they all map onto one engine path.
EXECUTE_MODES = ("torch", "hybrid parallel", "data parallel", "tree parallel", "auto")

#: Numerical slack for "the engine reproduces this hand-computed loss".
LOSS_TOL = 1e-6

# ---------------------------------------------------------------------------
# Hand-built specimen: (x0 - x2) * (x0 - x2)
# ---------------------------------------------------------------------------
# node_type base values: 0 VAR, 2 UFUNC, 3 BFUNC, so this prefix encoding is
#   node 0: BFUNC 3 (`*`)  -> children 1 and 4
#   node 1: BFUNC 2 (`-`)  -> children 2 and 3
#   node 2: VAR x0, node 3: VAR x2
#   node 4: BFUNC 2 (`-`)  -> children 5 and 6
#   node 5: VAR x0, node 6: VAR x2
# (node_value carries the variable index for VAR nodes and the `Func` id for
# function nodes; subtree_size[i] counts the nodes of the subtree rooted at i,
# and node 7 is the unused tail of the 8-slot buffer.)
tree = Tree(
    3,
    1,
    node_type=torch.tensor([3, 3, 0, 0, 3, 0, 0, 0], dtype=torch.int16, device=DEVICE),
    node_value=torch.tensor(
        [3.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0], dtype=torch.float32, device=DEVICE
    ),
    subtree_size=torch.tensor([7, 3, 1, 1, 3, 1, 1, 0], dtype=torch.int16, device=DEVICE),
)

#: The full 8-row XOR truth table; the third column is the constant-zero variable
#: x2 that the specimen reads (so the tree evaluates as x0**2 on this table).
XOR_INPUTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.float,
    device=DEVICE,
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    device=DEVICE,
)


def check(condition, message):
    """Assert ``condition``; the message is the check that failed."""
    if not condition:
        raise AssertionError(message)


def check_close(got, expected, message, tol=LOSS_TOL):
    check(abs(got - expected) <= tol, f"{message}: got {got!r}, expected {expected!r}")


def main():
    inputs, labels = XOR_INPUTS, XOR_OUTPUTS
    num_data = inputs.shape[0]

    # --- the specimen decodes to the expression it was built for -------------
    check(
        str(tree.to_sympy_expr()) == "(x0 - x2)**2",
        f"decoded expression changed: {tree.to_sympy_expr()!r} != '(x0 - x2)**2'",
    )

    # --- forward matches the closed form, batched and single row -------------
    expected_forward = (inputs[:, 0:1] - inputs[:, 2:3]) ** 2
    forward = tree.forward(inputs)
    check(
        forward.shape == expected_forward.shape,
        f"Tree.forward shape {tuple(forward.shape)} != {tuple(expected_forward.shape)}",
    )
    check(
        torch.equal(forward, expected_forward),
        f"Tree.forward != (x0 - x2)**2: {forward.reshape(-1).tolist()}",
    )
    for row in range(num_data):
        check(
            torch.equal(tree.forward(inputs[row]), expected_forward[row]),
            f"1-D Tree.forward disagrees with the batched result on row {row}",
        )

    # --- SR_fitness == loss recomputed from forward, for every execute_mode --
    diff = forward - labels
    expected_mse = float(diff.square().sum()) / num_data
    expected_mae = float(diff.abs().sum()) / num_data
    check_close(expected_mse, 0.5, "hand-computed MSE regression vector changed")
    check_close(expected_mae, 0.5, "hand-computed MAE regression vector changed")

    for use_mse, expected_loss in ((True, expected_mse), (False, expected_mae)):
        name = "MSE" if use_mse else "MAE"
        reference = None
        for mode in EXECUTE_MODES:
            got = tree.SR_fitness(inputs, labels, use_MSE=use_mse, execute_mode=mode)
            check(
                got.shape == (1,),
                f"Tree.SR_fitness({mode!r}) should return shape (1,), got {tuple(got.shape)}",
            )
            check_close(
                float(got[0]),
                expected_loss,
                f"{name} from Tree.SR_fitness(execute_mode={mode!r}) differs from Tree.forward",
            )
            # all modes share one implementation, so the tensors must be bit-equal
            if reference is None:
                reference = got
            else:
                check(
                    torch.equal(got, reference),
                    f"{name} is not bit-identical between execute_mode "
                    f"{EXECUTE_MODES[0]!r} and {mode!r}: {float(got[0])!r} vs {float(reference[0])!r}",
                )
        print(f"[ok] Tree.SR_fitness {name} = {expected_loss:.6f} on all {len(EXECUTE_MODES)} execute modes")

    # --- the batched wrapper agrees with the single-tree result --------------
    forest = tree.to_forest()
    check(forest.pop_size == 1, f"Tree.to_forest() should hold one tree, got {forest.pop_size}")
    forest_mse = forest.SR_fitness(inputs, labels)
    check(
        torch.equal(forest_mse, tree.SR_fitness(inputs, labels)),
        f"Forest.SR_fitness {float(forest_mse[0])!r} != Tree.SR_fitness "
        f"{float(tree.SR_fitness(inputs, labels)[0])!r} for a pop_size=1 forest",
    )
    check(
        torch.equal(forest[0].forward(inputs), forward),
        "Forest[0].forward disagrees with Tree.forward",
    )
    print("[ok] Tree.forward equals the decoded expression and Forest round-trips it")

    # --- an unknown execute_mode is rejected, not silently ignored -----------
    try:
        tree.SR_fitness(inputs, labels, execute_mode="data parallel ")
    except AssertionError:
        pass
    else:
        raise AssertionError("Tree.SR_fitness accepted an unknown execute_mode")

    print("all checks passed")


if __name__ == "__main__":
    main()
