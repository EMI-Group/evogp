import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.core import Tree, Forest, MAX_STACK
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)

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
    requires_grad=False,
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    requires_grad=False,
    device="cuda",
)


def evaluate(forest: Forest):
    loss = forest.SR_fitness(XOR_INPUTS, XOR_OUTPUTS)
    return -loss


generate_configs = {
    "const_prob": 0.5,
    "out_prob": 0.5,
    "func_prob": {"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25},
    "layer_leaf_prob": 0.2,
    "const_range": (-1, 1),
    "sample_cnt": 8,
}

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2, max_layer_cnt=3, **generate_configs),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=1000000,
    gp_len=512,
    input_len=3,
    output_len=1,
    max_layer_cnt=5,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = evaluate(forest)

for i in range(1000):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = evaluate(forest)
    toc = time.time()
    print(
        f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    )
