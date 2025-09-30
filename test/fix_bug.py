import torch
import evogp
from evogp.tree import Tree

tree = Tree(
    3,
    1,
    node_type=torch.tensor([3, 3, 0, 0, 3, 0, 0, 0], dtype=torch.int16, device="cuda"),
    node_value=torch.tensor([3., 2., 0., 2., 2., 0., 2., 0.], dtype=torch.float32,  device="cuda"),
    subtree_size=torch.tensor([7, 3, 1, 1, 3, 1, 1, 0], dtype=torch.int16, device="cuda"),
)

XOR_INPUTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        # [1, 0, 0],
        # [1, 0, 1],
        # [1, 1, 0],
        # [1, 1, 1],
    ],
    dtype=torch.float,
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [
        [0], 
        [1], 
        [1], 
        [0], 
        # [1], 
        # [0], 
        # [0], 
        # [1]
    ],
    dtype=torch.float,
    device="cuda",
)

fit = tree.SR_fitness(XOR_INPUTS, XOR_OUTPUTS, execute_mode="hybrid parallel")
print(fit)

fit = tree.SR_fitness(XOR_INPUTS, XOR_OUTPUTS, execute_mode="data parallel")
print(fit)