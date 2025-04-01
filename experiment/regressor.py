import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import SymbolicRegression
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    DeleteMutation,
    CombinedMutation,
    LeafBiasedCrossover,
    MultiConstMutation,
    MultiPointMutation,
)


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


problem = SymbolicRegression(
    func=func, num_inputs=2, num_data=100, lower_bounds=-5, upper_bounds=5
)

descriptor = GenerateDescriptor(
    max_tree_len=128,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=7,
    const_samples=[-1, 0, 1],
    layer_leaf_prob=0,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(
        pop_size=1000, descriptor=descriptor
    ),
    crossover=LeafBiasedCrossover(leaf_bias=0.8),
    mutation=CombinedMutation(
        [
            # DefaultMutation(
            #     mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
            # ),
            MultiConstMutation(
                mutation_rate=0.8, mutation_intensity=0.03, descriptor=descriptor
            ),
            MultiPointMutation(
                mutation_rate=0.8, mutation_intensity=0.03, descriptor=descriptor
            ),
            DeleteMutation(mutation_rate=0.95),
        ]
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    enable_pareto_front=True,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=300,
)

best = pipeline.run()

sympy_expression = best.to_sympy_expr()
print(sympy_expression)

print(algorithm.pareto_front)


import matplotlib.pyplot as plt
import numpy as np

fitness = algorithm.pareto_front.fitness.cpu()
complexity = np.arange(len(fitness))  # 复杂度范围

# 过滤掉适应度为 -inf 的数据
valid_complexity = []
valid_fitness = []

for c, f in zip(complexity, fitness):
    if f != -np.inf:
        valid_complexity.append(c)
        valid_fitness.append(f)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(valid_complexity, valid_fitness, label="Fitness", color="b")
plt.xlabel("Complexity")
plt.ylabel("Fitness")
plt.title("Pareto Front Fitness vs. Complexity")
plt.grid(True)
plt.legend()
plt.show()
