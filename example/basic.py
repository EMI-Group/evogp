import torch
from evogp.tree import Forest, GenerateDiscriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression
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

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS)

# create decriptor for generating new trees
descriptor = GenerateDiscriptor(
    max_tree_len=32,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=5000, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

pred_res = best.forward(XOR_INPUTS)
print(pred_res)

infix_expression = best.to_infix()
print(infix_expression)

# best.to_png("./imgs/xor_tree.png")