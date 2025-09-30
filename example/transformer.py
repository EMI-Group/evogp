import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Forest
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.tree import GenerateDescriptor
from evogp.problem import Transformation


def print_color(text):
    print(f"\033[33m{text}\033[0m")


transformation_problem = {
    "diabetes": {"input_len": 10},
}
name = list(transformation_problem.keys())[0]
print_color(f"Problem: {name}")

problem = Transformation(dataset=name)

descriptor = GenerateDescriptor(
    max_tree_len=128,
    input_len=transformation_problem[name]["input_len"],
    output_len=1,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)

algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

fitness = problem.evaluate(algorithm.forest)


from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

diabetes = load_diabetes()
estimator = Ridge()

# before transformation
estimator.fit(diabetes.data, diabetes.target)
print(estimator.score(diabetes.data, diabetes.target))

# transformation
for i in range(100):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    toc = time.time()
    print(f"step: {i}, max_fitness: {fitness.max()}")

new_diabetes = torch.cat(
    (problem.datapoints, problem.new_feature(forest, 100, 10)), dim=1
)

# after transformation
estimator.fit(new_diabetes.cpu().numpy(), diabetes.target)
print(estimator.score(new_diabetes.cpu().numpy(), diabetes.target))
