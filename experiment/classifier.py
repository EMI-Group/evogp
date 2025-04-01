import torch
import numpy as np

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
    SinglePointMutation,
)
from evogp.problem import Classification


dataset_name = ["iris", "wine", "breast_cancer", "digits"]


multi_output = True

problem = Classification(multi_output, dataset="iris")

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=CombinedMutation(
        [
            # DefaultMutation(
            #     mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
            # ),
            SinglePointMutation(
                mutation_rate=0.8, descriptor=descriptor, modify_output=True
            ),
        ]
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    enable_pareto_front=False,
)

for i in range(200):
    if i == 100:
        algorithm.mutation = CombinedMutation(
            [
                SinglePointMutation(
                    mutation_rate=0.8, descriptor=descriptor, modify_output=False
                ),
            ]
        )
    # evaluate
    fitness = problem.evaluate(algorithm.forest)
    torch.cuda.synchronize()

    # update
    algorithm.step(fitness)
    torch.cuda.synchronize()

    # record
    scores = fitness.cpu().numpy()
    valid_fitness = scores[~np.isinf(scores)]
    max_f, min_f, mean_f, std_f = (
        max(valid_fitness),
        min(valid_fitness),
        np.mean(valid_fitness),
        np.std(valid_fitness),
    )
    print(f"Generation {i}: max={max_f:.3f}, min={min_f:.3f}, mean={mean_f:.3f}")

best_idx = int(np.argmax(scores))
best = algorithm.forest[best_idx]
best.to_png("./imgs/classifier_tree.png")
sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# print(algorithm.pareto_front)
