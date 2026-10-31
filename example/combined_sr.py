"""Multi-part ("combined") symbolic regression on the 3-input XOR data set.

EvoGP's multi-output model is a *combined forest*: instead of one tree with several outputs,
every named variable of a model gets its own sub-forest over its own named inputs, described
by a ``data_info`` dict (``output_name -> [input_name, ...]``). The model is then a plain
Python function of those variables, so it is trained by handing it to
``evogp.problem.CustomLoss``: the loss function's ``existing_data`` parameters
(the data) stay shared while its remaining parameters receive the sub-forest outputs.
``CustomLoss`` is the supported route for a ``CombinedForest`` -- ``SymbolicRegression``
rejects one with ``NotImplementedError`` (its ``formula=``/``share_input=``-style API was
replaced by this dict protocol).

Here three sub-forests ``A``, ``B`` and ``C`` are evolved over the same three XOR inputs (the
"share input" case: all three read ``x0``, ``x1``, ``x2``) and are scored on ``A + B * C``
against the 3-input parity target. The GP loop is the combined counterpart of the single
``Forest`` loop in ``example/basic.py``: ``CombinedDefaultCrossover`` +
``CombinedDefaultMutation`` + ``DefaultSelection``, over one ``GenerateDescriptor`` that is
replicated for each sub-forest.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/combined_sr.py

As shipped (pop 100 x 3 sub-forests, 100 generations, ``max_tree_len=32``) the run takes
about 19 s on the reference box at the shipped default route (``auto`` -> the fused Triton
route), and the reported best fitness (``-(A + B * C - target)**2``, so 0.0 is a perfect
fit) goes from about -0.50 at generation 0 to about -0.22 at generation 100.
``EVOXC_DISABLE_LOCATIONS=1`` makes the evoxc tracing several times faster (tracing-only
knob, it does not change the fitness); ``EVOGP_EVOXC_BACKEND=iree`` switches evaluation to
the compiled IREE backend.
"""

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.tree import CombinedForest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    CombinedDefaultMutation,
    CombinedDefaultCrossover,
)
from evogp.problem import CustomLoss
from evogp.pipeline import StandardPipeline

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
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    device="cuda",
)

# The multi-part model: one sub-forest per variable of ``A + B * C``, each taking the three
# columns below as its inputs. ``CombinedForest.random_generate`` builds all sub-forests
# from the descriptors and checks input_len/output_len against this description.
data_info = {
    "A": ["x0", "x1", "x2"],
    "B": ["x0", "x1", "x2"],
    "C": ["x0", "x1", "x2"],
}

# The named inputs the loss function reads directly; every name that is NOT in this dict
# must be a sub-forest output of ``data_info``. ``CustomLoss`` passes the existing_data
# arguments as-is and maps over (vmap) the remaining ones, so list them first, in the order
# below, and reduce to one scalar per individual (here ``mean`` over the data points).
EXISTING_DATA = {
    "x0": XOR_INPUTS[:, 0],
    "x1": XOR_INPUTS[:, 1],
    "x2": XOR_INPUTS[:, 2],
    "target": XOR_OUTPUTS[:, 0],
}


def loss(x0, x1, x2, target, A, B, C):
    return ((A + B * C - target) ** 2).mean()


problem = CustomLoss(EXISTING_DATA, loss)

# one descriptor per sub-forest (a single descriptor is replicated automatically); every
# sub-forest here has 3 inputs and 1 output
descriptor = GenerateDescriptor(
    max_tree_len=32,
    input_len=len(data_info["A"]),
    output_len=1,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=CombinedForest.random_generate(
        pop_size=100,
        data_info=data_info,
        descriptors=descriptor,
    ),
    crossover=CombinedDefaultCrossover(),
    mutation=CombinedDefaultMutation(
        mutation_rate=0.2, descriptors=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

# evaluate the best individual on every XOR row: the combined containers take a dict of
# input-name -> values and return a dict of output-name -> values
data = {name: XOR_INPUTS[:, i] for i, name in enumerate(["x0", "x1", "x2"])}
pred_res = best.to_combined_forest().batch_forward(data)
parts = {name: res[0, :, 0] for name, res in pred_res.items()}
print("A:", parts["A"])
print("B:", parts["B"])
print("C:", parts["C"])
print("A + B * C:", parts["A"] + parts["B"] * parts["C"])

print(best.A.to_sympy_expr())
print(best.B.to_sympy_expr())
print(best.C.to_sympy_expr())

print(best.A.to_sympy_expr() + best.B.to_sympy_expr() * best.C.to_sympy_expr())
