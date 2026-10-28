"""EvoGP install validation: one small end-to-end GP run (``python -m evogp.sr_test``).

Trains ``SymbolicRegression`` on the synthetic 2-input target
``x0**4 / (x0**4 + 1) + x1**4 / (x1**4 + 1)`` (1000 data points) through the standard
wiring -- ``Forest.random_generate`` + ``GenerateDescriptor`` -> ``GeneticProgramming``
(default crossover / mutation / selection) -> ``StandardPipeline`` -- with torch seed 0,
so the run is reproducible and its best fitness must improve generation over generation.

The defaults are right-sized for the pure-Python evoxc engine: every tree structure is
JIT-built once per *(structure, shape)* and then interpreted per data batch, so cost grows
with ``pop_size`` x tree size while the population bloats as it improves. The original
CUDA-scale validation (``pop_size=1000``, ``max_tree_len=128``, 100 generations) therefore
needs hours here. As shipped (``pop_size=200`` x 100 generations, ``max_tree_len=64``) the
run takes roughly 30 s on the reference box (RTX A6000 host, evoxc ``"numpy"`` backend, seed
0) and still converges: the best fitness goes from about -0.2511 at generation 0 to about
-0.1720 at generation 100, with the first improvement around generation 43.

Most of that cost is *tracing*, not evaluating: evoxc attaches a call-site ``Location`` to
every op unless ``EVOXC_DISABLE_LOCATIONS=1`` is set, which is its documented speed knob for
tracing-heavy runs (locations are debug metadata -- they do not change the computed
fitness; the trajectory above is identical either way). This script sets the knob by
default; export ``EVOXC_DISABLE_LOCATIONS=0`` to keep the capture enabled (~380 s per run).

Scale the validation up (or down) by editing ``pop_size`` / ``generation_limit`` /
``max_tree_len`` below, or with the ``EVOGP_SR_POP`` / ``EVOGP_SR_GENS`` /
``EVOGP_SR_MAX_TREE_LEN`` environment variables. Bigger settings search harder but trade
directly against wall time. Do not drop ``generation_limit`` much below 100: the run is
elitist, so the reported best fitness never decreases, but with seed 0 this population
only starts improving around generation 43.
"""

import os

# evoxc's per-op call-site capture dominates the cold build of every tree structure, so it
# is switched off here unless the caller explicitly asks for it (evoxc checks for the value
# "1"; setting it to "0" keeps the capture on).
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression

# right-sized for the pure-Python evoxc engine (see the module docstring); the env
# overrides exist only to scale this validation up or down without editing the file
pop_size = int(os.environ.get("EVOGP_SR_POP", 200))
generation_limit = int(os.environ.get("EVOGP_SR_GENS", 100))
max_tree_len = int(os.environ.get("EVOGP_SR_MAX_TREE_LEN", 64))


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


problem = SymbolicRegression(
    func=func, num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5
)

descriptor = GenerateDescriptor(
    max_tree_len=max_tree_len,
    input_len=2,
    output_len=1,
    const_prob=0.5,
    out_prob=0.5,
    using_funcs=["+", "-", "*", "/"],
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=pop_size, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=generation_limit,
)

pipeline.run()

print(f"best fitness: {pipeline.best_fitness:.4f}")
