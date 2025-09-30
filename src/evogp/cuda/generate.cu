#include "kernel.h"
#include "defs.h"
#include <stdio.h>


/**
 * @brief CUDA kernel to generate a population of genetic programming (GP) trees.
 *
 * @tparam multiOutput Flag indicating if multi-output nodes should be generated.
 * @param results Output array of GP trees.
 * @param keys Random engine seed keys.
 * @param depth2leafProbs Array defining probability of generating leaf nodes based on depth.
 * @param rouletteFuncs Roulette wheel probabilities for selecting function nodes.
 * @param constSamples Array of constant values available for leaf nodes.
 */
template<bool multiOutput = false>
__global__ void treeGPGenerate(
	const unsigned int popSize,
	const unsigned int gpLen,
	const unsigned int varLen,
	const unsigned int outLen,
	const unsigned int constSamplesLen,
	const float outProb,
	const float constProb,
	float* value_res, 
	int16_t* type_res, 
	int16_t* subtree_size_res, 
	const unsigned int* keys, 
	const float* depth2leafProbs, 
	const float* rouletteFuncs, 
	const float* constSamples
)
{	
	// Calculate tree index
	const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// Initialize probabilities and random engine
	float leafProbs[MAX_FULL_DEPTH]{}, funcRoulette[Function::END]{};
	RandomEngine engine(hash(n, keys[0], keys[1]));
	thrust::uniform_real_distribution<float> rand(0.0f, 1.0f);

	// Load probabilities into local memory
	#pragma unroll
	for (int i = 0; i < MAX_FULL_DEPTH; i++)
	{
		leafProbs[i] = depth2leafProbs[i];
	}
	#pragma unroll
	for (int i = 0; i < Function::END; i++)
	{
		funcRoulette[i] = rouletteFuncs[i];
	}

	// stack memory for node generation
	GPNode* gp = (GPNode*)alloca(MAX_STACK * sizeof(GPNode));  // result gp array
	NchildDepth* childsAndDepth = (NchildDepth*)alloca(MAX_STACK * sizeof(NchildDepth));  // stack
	childsAndDepth[0] = { 1, 0 };  // Start with the root node, {child, depth}
	int topGP = 0, top = 1;

	// generate
	while (top > 0)
	{
		NchildDepth cd = childsAndDepth[--top];  // get one in stack
		cd.childs--;

		NchildDepth cdNew{};  // new childDepth
		GPNode node{.0f, (int16_t)(0), (int16_t)(0)};  //new node, {value, type, subtree_size}

		// Determine whether to generate a leaf or function node
		if (rand(engine) >= leafProbs[cd.depth])
		{	
			// generate non-leaf (function) node
			float r = rand(engine);
			int k = 0;  // function type
			#pragma unroll
			for (int i = Function::END - 1; i >= 0; i--)
			{
				if (r >= funcRoulette[i])
				{
					k = i + 1;
					break;
				}
			}
			int16_t type = k <= Function::IF ? NodeType::TFUNC : k <= Function::GE ? NodeType::BFUNC : NodeType::UFUNC;  // node type
			if constexpr (multiOutput)
			{
				if (rand(engine) <= outProb)
				{	
					// output node
					int16_t outType = type + NodeType::OUT_NODE;
					// value(float32) contains the function(int16_t) and outIndex(int16_t) info will using multiOutput mode
					OutNodeValue outNode{ (int16_t)k, (int16_t)(engine() % outLen) };  // {function type, idx}
					node = GPNode{ outNode, outType, 1 };  // subtreesize temporarily set to 1
				}
			}
			// node.subtreeSize == 0 means not multiOutput node
			if (node.subtreeSize == 0)
			{
				node = GPNode{ float(k), type, 1 };
			}
			cdNew = NchildDepth{ int16_t(type - 1), int16_t(cd.depth + 1) };  // {number of operants, depth + 1}
		}
		else
		{	
			// generate leaf node
			float value{};
			int16_t type{};
			if (rand(engine) <= constProb)
			{	
				// constant
				value = constSamples[engine() % constSamplesLen];
				type = NodeType::CONST;
			}
			else
			{	
				// variable
				value = engine() % varLen;
				type = NodeType::VAR;
			}
			node = GPNode{ value, type, 1 };    // subtreesize of a leaf is 1
		}
		gp[topGP++] = node;  // add node in res_gp
		if (cd.childs > 0)  // still has child to add
			childsAndDepth[top++] = cd;  // add its child to stack
		if (cdNew.childs > 0)
			childsAndDepth[top++] = cdNew;  //add child's child to stack
	}

	// Calculate subtree sizes
	int* nodeSize = (int*)childsAndDepth;  // reuse space
	top = 0;
	for (int i = topGP - 1; i >= 0; i--)
	{
		int16_t node_type = gp[i].nodeType;
		node_type &= NodeType::TYPE_MASK;
		if (node_type <= NodeType::CONST)  // VAR or CONST
		{
			nodeSize[top] = 1;
		}
		else if (node_type == NodeType::UFUNC)
		{
			int size1 = nodeSize[--top];
			nodeSize[top] = size1 + 1;
		}
		else if (node_type == NodeType::BFUNC)
		{
			int size1 = nodeSize[--top], size2 = nodeSize[--top];
			nodeSize[top] = size1 + size2 + 1;
		}
		else // if (node_type == NodeType::TFUNC)
		{
			int size1 = nodeSize[--top], size2 = nodeSize[--top], size3 = nodeSize[--top];
			nodeSize[top] = size1 + size2 + size3 + 1;
		}
		gp[i].subtreeSize = (int16_t)nodeSize[top];
		top++;
	}

	// Write result to global memory
	const int len = gp[0].subtreeSize;

	auto o_value = value_res + n * gpLen;
	auto o_type = type_res + n * gpLen;
	auto o_subtree_size = subtree_size_res + n * gpLen;

	for (int i = 0; i < len; i++)
	{
		o_value[i] = gp[i].value;
		o_type[i] = gp[i].nodeType;
		o_subtree_size[i] = gp[i].subtreeSize;
	}
}


/**
 * @brief Launch the kernel to generate GP trees.
 */
template<bool MultiOutput = false>
inline void generateExecuteKernel(
	const unsigned int popSize,
	const unsigned int maxGPLen,
	const unsigned int varLen,
	const unsigned int outLen,
	const unsigned int constSamplesLen,
	const float outProb,
	const float constProb,
	const unsigned int* keys, 
	const float* depth2leafProbs, 
	const float* rouletteFuncs, 
	const float* constSamples, 
	float* value_res, 
	int16_t* type_res, 
	int16_t* subtree_size_res
)
{	
	int gridSize = 0, blockSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPGenerate<MultiOutput>);
	if (gridSize * blockSize < popSize)
	{
		gridSize = (popSize - 1) / blockSize + 1;
	}
	treeGPGenerate<MultiOutput><<<gridSize, blockSize>>>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, value_res, type_res, subtree_size_res, keys, depth2leafProbs, rouletteFuncs, constSamples);
}


/**
 * @brief Public function to generate GP trees based on the provided descriptor. Handle data type selection for the GP generation process.
 */
void generate(
	const unsigned int popSize,
	const unsigned int maxGPLen,
	const unsigned int varLen,
	const unsigned int outLen,
	const unsigned int constSamplesLen,
	const float outProb,
	const float constProb,
	const unsigned int* keys, 
	const float* depth2leafProbs, 
	const float* rouletteFuncs, 
	const float* constSamples, 
	float* value_res, 
	int16_t* type_res, 
	int16_t* subtree_size_res
	){
	if (outLen > 1)
	{
		generateExecuteKernel<true>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, keys, depth2leafProbs, rouletteFuncs, constSamples, value_res, type_res, subtree_size_res);
	}
	else
	{
		generateExecuteKernel<false>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, keys, depth2leafProbs, rouletteFuncs, constSamples, value_res, type_res, subtree_size_res);
	}
}