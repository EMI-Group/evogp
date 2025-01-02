import torch
import numpy as np
import pandas as pd

import sys

sys.argv.append("100")  # popsize
sys.argv.append("8")  # datapoint
sys.argv.append("1")  # time of run
pop_size = int(sys.argv[1])
datapoint = int(sys.argv[2])
random_seed = int(sys.argv[3])

torch.random.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

import time
from evogp.core import Tree, Forest, MAX_STACK
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    TournamentSelection,
)


def update_csv(file_path, pop_size, datapoint, mse, time):
    df = pd.read_csv(file_path, index_col="pop")

    mse_column = f"{datapoint}^2 MSE"
    time_column = f"{datapoint}^2 Time"

    if pop_size in df.index:
        df.at[pop_size, mse_column] = mse
        df.at[pop_size, time_column] = time
    else:
        new_row = pd.DataFrame(
            {mse_column: [mse], time_column: [time]}, index=[pop_size]
        )
        df = pd.concat([df, new_row])

    df.to_csv(file_path, index_label="pop")


a = torch.empty(datapoint**2, 1, dtype=torch.float, device="cuda").uniform_(-5, 5)
b = torch.empty(datapoint**2, 1, dtype=torch.float, device="cuda").uniform_(-5, 5)
INPUTS = torch.cat((a, b), dim=1)


@torch.vmap
def my_function(x):
    return x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)


OUTPUTS = my_function(INPUTS).reshape(-1, 1)


def evaluate(forest: Forest):
    loss = forest.SR_fitness(INPUTS, OUTPUTS)
    return -loss


generate_configs = {
    "gp_len": 400,
    "input_len": 2,
    "output_len": 1,
    "out_prob": 0.0,
    "const_prob": 0.5,
    "func_prob": {"+": 3, "-": 3, "*": 3, "/": 3, "sin": 1, "cos": 1, "tan": 1},
    "const_samples": torch.tensor(
        [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="cuda"
    ),
}

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        depth2leaf_probs=torch.tensor(
            [0, 0, 0, 0.3, 0.7, 1, 1, 1, 1, 1], dtype=torch.float32, device="cuda"
        ),
        **generate_configs,
    ),
    selection=TournamentSelection(20, 0.9, False),
)

# initialize the forest
forest = algorithm.initialize(
    pop_size,
    depth2leaf_probs=torch.tensor([0, 0, 0, 0, 0.2, 0.4, 0.8, 1, 1, 1], device="cuda"),
    **generate_configs,
)
fitness = evaluate(forest)
# print(fitness)

start_time = time.time()
for i in range(50):
    print(i)
    forest = algorithm.step(fitness)
    fitness = evaluate(forest)
end_time = time.time()

print(
    f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {end_time - start_time}"
)
try:
    with open(f"len400_{sys.argv[3]}.csv", "x") as file:
        file.write("pop")
except:
    pass
update_csv(
    f"len400_{sys.argv[3]}.csv",
    pop_size,
    datapoint,
    -fitness.max().item(),
    end_time - start_time,
)
