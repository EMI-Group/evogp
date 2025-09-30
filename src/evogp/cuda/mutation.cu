#include "kernel.h"
#include "defs.h"
#include <stdio.h>

__host__ __device__
inline void _gpTreeReplace(
	const int old_node_idx, 
	const int new_node_idx, 
	const int new_subsize, 
	const int old_offset, 
	const int old_size, 
	const int size_diff, 
    const float* value_old,
    const int16_t* type_old,
    const int16_t* subtree_size_old,
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res
)
{
	// create stack memory
	float* value_stack = (float*)alloca(MAX_STACK * sizeof(float));
	int16_t* type_stack = (int16_t*)alloca(MAX_STACK * sizeof(int16_t));
	int16_t* subtree_size_stack = (int16_t*)alloca(MAX_STACK * sizeof(int16_t));

	// copy previous part
	for (int i = 0; i < old_node_idx; i++)
	{	
		value_stack[i] = value_old[i];
		type_stack[i] = type_old[i];
		subtree_size_stack[i] = subtree_size_old[i];
	}

	// change subtree sizes of ancestors
	int current = 0;
	while (current < old_node_idx)
	{
		int midTreeIndex{}, rightTreeIndex{};
		subtree_size_stack[current] += size_diff;
		auto node_type = type_stack[current];
		node_type &= NodeType::TYPE_MASK;
		current++;

		if (current >= old_node_idx){  // actually, current == old_node_idx
			break;
		}

		switch (node_type)
		{
		case NodeType::UFUNC:
			// do nothing
			break;
		case NodeType::BFUNC:
			rightTreeIndex = current + subtree_size_stack[current];
			if (old_node_idx < rightTreeIndex)
			{	// at left subtree
				// do nothing
			}
			else
			{	// at right subtree
				current = rightTreeIndex;
			}
			break;
		case NodeType::TFUNC:
			midTreeIndex = subtree_size_stack[current] + current;
			if (old_node_idx < midTreeIndex)
			{	
				// at left subtree
				// do nothing
				break;
			}
			rightTreeIndex = subtree_size_stack[midTreeIndex] + midTreeIndex;
			if (old_node_idx < rightTreeIndex)  // midTreeIndex <= old_node_idx &&
			{	// at mid subtree
				current = midTreeIndex;
			}
			else
			{	// at right subtree
				current = rightTreeIndex;
			}
			break;
		default: // must the subtree itself
			break;
		}
	}
	
	// copy new tree
	for (int i = 0; i < new_subsize; i++)
	{	
		value_stack[i + old_node_idx] = value_new[i + new_node_idx];
		type_stack[i + old_node_idx] = type_new[i + new_node_idx];
		subtree_size_stack[i + old_node_idx] = subtree_size_new[i + new_node_idx];
	}

	// copy remain old tree
	for (int i = old_offset; i < old_size; i++)
	{	
		value_stack[i + size_diff] = value_old[i];
		type_stack[i + size_diff] = type_old[i];
		subtree_size_stack[i + size_diff] = subtree_size_old[i];
	}

	// copy result from stack memory to global memory
	const int len = subtree_size_stack[0];
	for (int i = 0; i < len; i++)
	{
		value_res[i] = value_stack[i];
		type_res[i] = type_stack[i];
		subtree_size_res[i] = subtree_size_stack[i];
	}

}


__global__ void treeGPMutationKernel(
    const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
	const int* mutateIndices, 
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res,
	const int popSize,
	const int maxGPLen
)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	
	// output location
	auto out_value = value_res + n * maxGPLen;
	auto out_type = type_res + n * maxGPLen;
	auto out_subsize = subtree_size_res + n * maxGPLen;

	// origin location
	auto old_value = value_ori + n * maxGPLen;
	auto old_type = type_ori + n * maxGPLen;
	auto old_subsize = subtree_size_ori + n * maxGPLen;

	const int old_node_idx = mutateIndices[n];
	const int old_size = old_subsize[0];

	if (old_node_idx < 0 || old_node_idx >= old_size)
	{	// invalid node index
		// (do nothing) copy old tree to output
		for (int i = 0; i < old_size; i++)
		{	
			out_value[i] = old_value[i];
			out_type[i] = old_type[i];
			out_subsize[i] = old_subsize[i];
		}
		return;
	}

	// new tree location
	auto new_value = value_new + n * maxGPLen;
	auto new_type = type_new + n * maxGPLen;
	auto new_subsize = subtree_size_new + n * maxGPLen;

	const int oldSubtreeSize = old_subsize[old_node_idx], newSubtreeSize = new_subsize[0];
	const int sizeDiff = newSubtreeSize - oldSubtreeSize;
	const int oldOffset = old_node_idx + oldSubtreeSize;
	if (old_size + sizeDiff > maxGPLen)
	{	// too large output size
		// (do nothing) copy old tree to output
		for (int i = 0; i < old_size; i++)
		{	
			out_value[i] = old_value[i];
			out_type[i] = old_type[i];
			out_subsize[i] = old_subsize[i];
		}
		return;
	}

	// excute replace
	_gpTreeReplace(old_node_idx, 0, newSubtreeSize, oldOffset, old_size, sizeDiff, old_value, old_type, old_subsize, new_value, new_type, new_subsize, out_value, out_type, out_subsize);
}

void mutate(
    int popSize,
	int gpLen,
    const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
    const int* mutateIndices, 
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res
    )
{
    int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPMutationKernel);
	if (gridSize * blockSize < popSize)
		gridSize = (popSize - 1) / blockSize + 1;
	treeGPMutationKernel<<<gridSize, blockSize>>>(
		value_ori, 
		type_ori, 
		subtree_size_ori, 
		mutateIndices, 
		value_new, 
		type_new, 
		subtree_size_new, 
		value_res, 
		type_res, 
		subtree_size_res, 
		popSize, 
		gpLen
	);
}




__global__ void treeGPCrossoverKernel(
	const int pop_size_ori,
	const int pop_size_new, 
	const int maxGPLen,
	const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
	const int* left_idx, 
	const int* right_idx, 
	const int* left_node_idx, 
	const int* right_node_idx, 
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res
)
{	
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= pop_size_new)
		return;

	// output location
	auto out_value = value_res + n * maxGPLen;
	auto out_type = type_res + n * maxGPLen;
	auto out_subsize = subtree_size_res + n * maxGPLen;

	// left tree location
	auto left_value = value_ori + left_idx[n] * maxGPLen;
	auto left_type = type_ori + left_idx[n] * maxGPLen;
	auto left_subsize = subtree_size_ori + left_idx[n] * maxGPLen;

	const int left_size = left_subsize[0];

	if (right_idx[n] < 0 || right_idx[n] >= pop_size_ori)
	{	// invalid right
		for (int i = 0; i < left_size; i++)
		{	
			// directly copy left
			out_value[i] = left_value[i];
			out_type[i] = left_type[i];
			out_subsize[i] = left_subsize[i];
		}
		return;
	}

	// right tree location
	auto right_value = value_ori + right_idx[n] * maxGPLen;
	auto right_type = type_ori + right_idx[n] * maxGPLen;
	auto right_subsize = subtree_size_ori + right_idx[n] * maxGPLen;

	const int left_node_size = left_subsize[left_node_idx[n]];
	const int right_node_size = right_subsize[right_node_idx[n]];

	const int size_diff = right_node_size - left_node_size;
	const int left_offset = left_node_idx[n] + left_node_size;

	if (left_size + size_diff > maxGPLen)
	{	// too large output size
		for (int i = 0; i < left_size; i++)
		{
			// directly copy left
			out_value[i] = left_value[i];
			out_type[i] = left_type[i];
			out_subsize[i] = left_subsize[i];
		}
		return;
	}

	// replace
	_gpTreeReplace(
		left_node_idx[n], 
		right_node_idx[n], 
		right_node_size, 
		left_offset, 
		left_size, 
		size_diff, 
		left_value,
		left_type,
		left_subsize,
		right_value,
		right_type,
		right_subsize,
		out_value,
		out_type,
		out_subsize
	);
}


void crossover(
	const int pop_size_ori,
	const int pop_size_new, 
	const int gpLen, 
	const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
	const int* left_idx, 
	const int* right_idx, 
	const int* left_node_idx, 
	const int* right_node_idx, 
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res
)
{
	int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPCrossoverKernel);
	if (gridSize * blockSize < pop_size_new)
		gridSize = (pop_size_new - 1) / blockSize + 1;
	treeGPCrossoverKernel<<<gridSize, blockSize>>>(
		pop_size_ori,
		pop_size_new,
		gpLen,
		value_ori,
		type_ori,
		subtree_size_ori,
		left_idx,
		right_idx,
		left_node_idx,
		right_node_idx,
		value_res,
		type_res,
		subtree_size_res
	);
}