import torch
from evogp.tree import CombinedForest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    CombinedDefaultMutation,
    CombinedDefaultCrossover,
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

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS, execute_mode="torch")

# create decriptor for generating new trees
descriptor = GenerateDescriptor(
    max_tree_len=32,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=CombinedForest.random_generate(
        formula=lambda A, B, C: A + B * C,
        pop_size=5000,
        descriptors=descriptor,
        share_input=True,
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

pred_res = best.forward(XOR_INPUTS)
print(pred_res)

print(best.A.to_sympy_expr())
print(best.B.to_sympy_expr())
print(best.C.to_sympy_expr())

print(best.to_sympy_expr(lambda A, B, C: A + B * C))
