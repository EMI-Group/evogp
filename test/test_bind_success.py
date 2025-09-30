import torch
import evogp.evogp_cuda
from evogp.tree.utils import str_tree
import numpy as np

pop_size = 10
gp_len = 1024
var_len = 2
out_len = 1
const_samples_len = 3
out_prob = 0.3
const_prob = 0.5


def generate(
    pop_size=10,
    gp_len=64,
    max_len=8,
    var_len=2,
    out_len=1,
    const_samples_len=3,
    out_prob=0.3,
    const_prob=0.5,
):

    max_layer_cnt = int(np.log2(max_len)) - 1

    keys = torch.tensor([42, 0], dtype=torch.uint32, device="cuda")
    depth2leaf_probs = torch.tensor(
        [0.1] * max_layer_cnt + [1.0] * (10 - max_layer_cnt),
        dtype=torch.float32,
        device="cuda",
    )
    roulette_funcs = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0] + [1.0] * 19, dtype=torch.float32, device="cuda"
    )  # only use +-*/
    const_samples = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cuda")

    value, node_type, subtree_size = torch.ops.evogp_cuda.tree_generate(
        pop_size,
        gp_len,
        var_len,
        out_len,
        const_samples_len,
        out_prob,
        const_prob,
        keys,
        depth2leaf_probs,
        roulette_funcs,
        const_samples,
    )

    return value, node_type, subtree_size


def mutate(
    value_ori,
    type_ori,
    subtree_size_ori,
    mutateIndices,
    value_new,
    type_new,
    subtree_size_new,
    pop_size=10,
    gp_len=64,
):
    return torch.ops.evogp_cuda.tree_mutate(
        pop_size,
        gp_len,
        value_ori,
        type_ori,
        subtree_size_ori,
        mutateIndices,
        value_new,
        type_new,
        subtree_size_new,
    )


def crossover(
    pop_size_ori,
    pop_size_new,
    gp_len,
    value_ori,
    type_ori,
    subtree_size_ori,
    left_idx,
    right_idx,
    left_node_idx,
    right_node_idx,
):
    return torch.ops.evogp_cuda.tree_crossover(
        pop_size_ori,
        pop_size_new,
        gp_len,
        value_ori,
        type_ori,
        subtree_size_ori,
        left_idx,
        right_idx,
        left_node_idx,
        right_node_idx,
    )


def evaluate(
    pop_size, gp_len, var_len, out_len, value, tree_type, subtree_size, variables
):
    return torch.ops.evogp_cuda.tree_evaluate(
        pop_size, gp_len, var_len, out_len, value, tree_type, subtree_size, variables
    )


def SR_fitnesses(
    pop_size,
    data_points,
    gp_len,
    var_len,
    out_len,
    use_MSE,
    value,
    tree_type,
    subtree_size,
    variables,
    labels,
):
    return torch.ops.evogp_cuda.tree_SR_fitness(
        pop_size,
        data_points,
        gp_len,
        var_len,
        out_len,
        use_MSE,
        value,
        tree_type,
        subtree_size,
        variables,
        labels,
        0
    )


def mutate_test():
    value, node_type, subtree_size = generate(pop_size=1, max_len=16)
    print(str_tree(value[0], node_type[0], subtree_size[0]))
    new_value, new_node_type, new_subtree_size = generate(pop_size=1, max_len=8)
    print(str_tree(new_value[0], new_node_type[0], new_subtree_size[0]))

    mutate_idx = torch.tensor([3], dtype=torch.int).cuda()
    mutated_value, mutated_type, mutated_subtree_size = mutate(
        value,
        node_type,
        subtree_size,
        mutate_idx,
        new_value,
        new_node_type,
        new_subtree_size,
        pop_size=1,
    )
    print(mutated_value, mutated_type, mutated_subtree_size)
    print(str_tree(mutated_value[0], mutated_type[0], mutated_subtree_size[0]))


def crossover_test():
    value, node_type, subtree_size = generate(pop_size=2, max_len=8, gp_len=64)
    print(str_tree(value[0], node_type[0], subtree_size[0]))
    print(str_tree(value[1], node_type[1], subtree_size[1]))

    left_idx = torch.tensor([0], dtype=torch.int).cuda()
    right_idx = torch.tensor([1], dtype=torch.int).cuda()
    left_node_idx = torch.tensor([2], dtype=torch.int).cuda()
    right_node_idx = torch.tensor([4], dtype=torch.int).cuda()

    new_value, new_node_type, new_subtree_size = crossover(
        2,
        1,
        64,
        value,
        node_type,
        subtree_size,
        left_idx,
        right_idx,
        left_node_idx,
        right_node_idx,
    )
    print(new_value, new_node_type, new_subtree_size)
    print(str_tree(new_value[0], new_node_type[0], new_subtree_size[0]))


def evaluate_test():
    value, node_type, subtree_size = generate(
        pop_size=2, var_len=2, out_len=1, max_len=8, gp_len=64
    )
    print(str_tree(value[0], node_type[0], subtree_size[0]))
    print(str_tree(value[1], node_type[1], subtree_size[1]))

    variables = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda()
    res = evaluate(2, 64, 2, 1, value, node_type, subtree_size, variables)
    print(res)


def SR_fitnesses_test():
    value, node_type, subtree_size = generate(
        pop_size=2, var_len=1, out_len=1, max_len=8, gp_len=64
    )
    print(str_tree(value[0], node_type[0], subtree_size[0]))
    print(str_tree(value[1], node_type[1], subtree_size[1]))

    variables = torch.tensor([[1], [1]], dtype=torch.float32).cuda()
    res = evaluate(2, 64, 1, 1, value, node_type, subtree_size, variables)
    print(res)

    data_points = torch.tensor([[1], [2]], dtype=torch.float32).cuda()
    labels = torch.tensor([[1], [3]], dtype=torch.float32).cuda()
    res = SR_fitnesses(
        2, 2, 64, 1, 1, True, value, node_type, subtree_size, data_points, labels
    )
    print(res)


def main():

    value, node_type, subtree_size = generate(
        pop_size=2, var_len=2, out_len=1, max_len=8, gp_len=64
    )
    print(str_tree(value[0], node_type[0], subtree_size[0]))

    crossover_test()
    evaluate_test()
    SR_fitnesses_test()


if __name__ == "__main__":
    main()
