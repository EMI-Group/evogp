import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evogp.cuda.operations import forward, sr_fitness

target_name = "Smax"

file_path = '/home/kelvin/data/x-t.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

input = data.iloc[:, :15].values.astype(np.float32)
trees = np.load(f'/home/kelvin/data/new_prob_{target_name}.npy')
ground_truth = data[target_name].values

# fitness = sr_fitness(trees, input, ground_truth)
# best_idx = np.argmin(fitness)
# print(best_idx)
# best_individual = trees[best_idx]

# best_individual = trees[0] # Katt
# best_individual = trees[1] # Kstr
best_individual = trees[111630] # Smax
predict = forward(np.tile(best_individual, (848, 1)), input[:848])

plt.figure(figsize=(10, 6))
plt.scatter(range(100), predict[:100], color='blue', marker='o')
plt.scatter(range(100), ground_truth[:100], color='red', marker='o')
plt.legend(['Predict', 'Ground Truth'])
plt.xlabel('Index')
plt.ylabel('y')
plt.grid(True)
plt.show()