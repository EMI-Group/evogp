import torch
from torch import Tensor
from ...tree import Forest


def subtensor(tensor: Tensor, start: Tensor, length: Tensor):
    """
    Extract a sub-tensor from the given tensor, starting at the given position
    and with the specified length.

    Args:
        tensor (Tensor): The original tensor to extract the sub-tensor from.
        start (Tensor): The start position of the sub-tensor, shape (batch_size, 1).
        length (Tensor): The length of the sub-tensor, shape (batch_size, 1).

    Returns:
        Tensor: The extracted sub-tensor, with the same shape as the original tensor.
    """
    start = start.reshape(-1, 1)
    length = length.reshape(-1, 1)
    end = start + length
    indices = torch.arange(tensor.shape[1], device="cuda") + start
    indices = torch.clamp(indices, max=tensor.shape[1] - 1)
    shifted_tensor = tensor.gather(1, indices)
    return torch.where(indices < end, shifted_tensor, 0)


def vmap_subtree(forest: Forest, pos: Tensor):
    """
    Extract a subtree from the forest at the given positions. The subtree includes 
    node values, types, and subtree sizes at the specified positions.
    
    Args:
        forest (Forest): The current forest object containing multiple trees.
        pos (Tensor): The positions of the subtrees to be extracted, shape (batch_size,).
    
    Returns:
        Forest: A new forest object containing the extracted subtrees.
    """
    pos = pos.reshape(-1, 1)
    length = forest.batch_subtree_size.gather(1, pos)
    return Forest(
        forest.input_len,
        forest.output_len,
        subtensor(forest.batch_node_value, pos, length),
        subtensor(forest.batch_node_type, pos, length),
        subtensor(forest.batch_subtree_size, pos, length),
    )
