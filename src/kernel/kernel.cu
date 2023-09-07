#include "kernel.h"
#include <malloc.h>

#ifdef TEST
#include <chrono> // for std::chrono functions

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_type = std::chrono::steady_clock;
	using second_type = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_type> m_beg;

public:
	Timer() : m_beg{ clock_type::now() }
	{
	}

	void reset()
	{
		m_beg = clock_type::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
	}
};
#endif


template<typename T>
__global__ void treeGPEvalKernel(const NodeInfo* nodeInfos, const T* prefixGPs, const T* variables, T* results, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int varLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	T* stack = (T*)alloca(MAX_STACK * sizeof(T));
	unsigned int* infos = (unsigned int*)alloca(MAX_STACK * sizeof(unsigned int));
	auto d_nodeInfos = nodeInfos + n * maxGPLen;
	auto d_nodeValue = prefixGPs + n * maxGPLen;
	auto d_vars = variables + n * varLen;
	auto s_vars = stack + MAX_STACK - varLen;
	unsigned int len = d_nodeInfos[0].subtreeSize;
	for (int i = 0; i < varLen; i++)
	{
		s_vars[i] = d_vars[i];
	}
	for (int i = 0; i < len; i++)
	{
		stack[len - i - 1] = d_nodeValue[i];
		infos[len - i - 1] = d_nodeInfos[i].nodeType;
	}
	// do stack operation according to the type of each node
	int top = 0;
	for (int i = 0; i < len; i++) {
		unsigned int node_type = infos[i];
		T node_value = stack[i];

		if (node_type == NodeType::CONST) {
			stack[top] = node_value;
			top++;
		}
		else if (node_type == NodeType::VAR) {
			int var_num = (int)node_value;
			stack[top] = s_vars[var_num];
			top++;
		}
		else if (node_type == NodeType::UFUNC) {
			int function = (int)node_value;
			top--;
			T var1 = stack[top];
			if (function == Function::SIN) {
				if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>)
					stack[top] = hsin(var1);
				else
					stack[top] = std::sin(var1);
				top++;
			}
			else if (function == Function::COS) {
				if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>)
					stack[top] = hcos(var1);
				else
					stack[top] = std::cos(var1);
				top++;
			}
			else if (function == Function::TAN) {
				if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>)
					stack[top] = hsin(var1) / hcos(var1);
				else
					stack[top] = std::tan(var1);
				top++;
			}
			else if (function == Function::LOG) {
				if (var1 <= T(0.0f)) {
					stack[top] = LOG_NEG;
					top++;
				}
				else {
					if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>)
						stack[top] = hlog(var1);
					else
						stack[top] = std::log(var1);
					top++;
				}
			}
			else if (function == Function::INV) {
				if (var1 == T(0.0f)) {
					var1 = DELTA;
				}
				stack[top] = T(1.0f) / var1;
				top++;
			}
			else if (function == Function::EXP) {
				if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>)
					stack[top] = hexp(var1);
				else
					stack[top] = std::exp(var1);
				top++;
			}
			else if (function == Function::NEG) {
				stack[top] = -var1;
				top++;
			}
		}
		else // if (node_type == NodeType::BFUNC)
		{
			int function = (int)node_value;
			top--;
			T var1 = stack[top];
			top--;
			T var2 = stack[top];
			if (function == Function::ADD) {
				stack[top] = var1 + var2;
				top++;
			}
			else if (function == Function::SUB) {
				stack[top] = var1 - var2;
				top++;
			}
			else if (function == Function::MUL) {
				stack[top] = var1 * var2;
				top++;
			}
			else if (function == Function::DIV) {
				if (var2 == T(0.0f)) {
					var2 = DELTA;
				}
				stack[top] = var1 / var2;
				top++;
			}
			else if (function == Function::MAX) {
				stack[top] = var1 >= var2 ? var1 : var2;
				top++;
			}
			else if (function == Function::MIN) {
				stack[top] = var1 <= var2 ? var1 : var2;
				top++;
			}
		}
	}
	// final
	results[n] = stack[--top];
}

template<typename T>
inline void eval_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const NodeInfo* nodeInfos, const void* prefixGPs, const void* variables, void* results)
{
	treeGPEvalKernel<T><<<(d.popSize - 1) / 128 + 1, 128, 0, stream>>>(nodeInfos, (const T*)prefixGPs, (const T*)variables, (T*)results, d.popSize, d.gpLen, d.varLen);
}

void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	// buffers: NodeInfo nodeInfos[popSize, gpLen], T prefixGPs[popSize, gpLen], T variables[popSize, varLen], T results[popSize]
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const NodeInfo* nodeInfos = (const NodeInfo*)(buffers[0]);
	const void* prefixGPs = (const void*)(buffers[1]);
	const void* variables = (const void*)(buffers[2]);
	void* results = (void*)(buffers[3]);
#ifdef TEST
	Timer t;
#endif
	switch (d.type)
	{
	case ElementType::BF16:
		eval_kernel<nv_bfloat16>(stream, d, nodeInfos, prefixGPs, variables, results);
		break;
	case ElementType::F16:
		eval_kernel<half>(stream, d, nodeInfos, prefixGPs, variables, results);
		break;
	case ElementType::F32:
		eval_kernel<float>(stream, d, nodeInfos, prefixGPs, variables, results);
		break;
	case ElementType::F64:
		eval_kernel<double>(stream, d, nodeInfos, prefixGPs, variables, results);
		break;
	default:
		throw std::runtime_error("Unsupported data type");
		break;
	}
#ifdef TEST
	auto err = cudaDeviceSynchronize();
	if (err != 0)
		throw std::runtime_error("Execution error of code " + (int)err);
	std::cout << t.elapsed() << std::endl;
#endif
}


template<typename T>
__global__ void treeGPCrossoverKernel(const NodeInfo* nodeInfos, const T* prefixGPs, NodeInfo* outInfos, T* outGPs, const int* leftPerms, const int* rightPerms, const UInt16_2* lrNodes, const unsigned int popSize, const unsigned int maxGPLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	T* gp = (T*)alloca(MAX_STACK * sizeof(T));
	NodeInfo* info = (NodeInfo*)alloca(MAX_STACK * sizeof(NodeInfo));
	auto d_leftInfo = nodeInfos + leftPerms[n] * maxGPLen;
	auto d_leftValue = prefixGPs + leftPerms[n] * maxGPLen;
	auto d_rightInfo = nodeInfos + rightPerms[n] * maxGPLen;
	auto d_rightValue = prefixGPs + rightPerms[n] * maxGPLen;
	const unsigned int leftNode = lrNodes[n].left, rightNode = lrNodes[n].right;
	const unsigned int oldSubtreeSize = d_leftInfo[leftNode].subtreeSize, newSubtreeSize = d_rightInfo[rightNode].subtreeSize;
	const unsigned int sizeDiff = newSubtreeSize - oldSubtreeSize, remainSize = d_leftInfo[0].subtreeSize - (leftNode + oldSubtreeSize);
	const unsigned int oldOffset = leftNode + oldSubtreeSize, newOffset = leftNode + newSubtreeSize;
	// copy
	for (int i = 0; i < leftNode; i++)
	{
		gp[i] = d_leftValue[i];
		info[i] = d_leftInfo[i];
	}
	for (int i = 0; i < newSubtreeSize; i++)
	{
		gp[i + leftNode] = d_rightValue[i + rightNode];
		info[i + leftNode] = d_rightInfo[i + rightNode];
	}
	for (int i = 0; i < remainSize; i++)
	{
		gp[i + newOffset] = d_leftValue[i + oldOffset];
		info[i + newOffset] = d_leftInfo[i + oldOffset];
	}
	// change subtree sizes of ancestors
	unsigned int current = 0;
	while (leftNode != current)
	{
		info[current].subtreeSize += sizeDiff;
		if (info[current + 1].subtreeSize + current + 1 > leftNode)
		{	// at left subtree
			current += 1;
		}
		else
		{	// at right subtree
			current += info[current + 1].subtreeSize + 1;
		}
	}
	// outupt
	const unsigned int len = info[0].subtreeSize;
	for (int i = 0; i < len; i++)
	{
		outGPs[i] = gp[i];
		outInfos[i] = info[i];
	}
}

template<typename T>
inline void crossover_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const NodeInfo* nodeInfos, const void* prefixGPs, NodeInfo* outInfos, void* outGPs, const int* leftPerms, const int* rightPerms, const UInt16_2* lrNodes)
{
	treeGPCrossoverKernel<T><<<(d.popSize - 1) / 128 + 1, 128, 0, stream>>>(nodeInfos, (const T*)prefixGPs, outInfos, (T*)outGPs, leftPerms, rightPerms, lrNodes, d.popSize, d.gpLen);
}

void treeGP_crossover(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	// buffers: NodeInfo* nodeInfos, T* prefixGPs, NodeInfo* outInfos, T* outGPs, int* leftPerms, int* rightPerms, uint16[2]* lrNodes
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const NodeInfo* nodeInfos = (const NodeInfo*)(buffers[0]);
	const void* prefixGPs = (const void*)(buffers[1]);
	const int* leftPerms = (const int*)(buffers[2]);
	const int* rightPerms = (const int*)(buffers[3]);
	const UInt16_2* lrNodes = (const UInt16_2*)(buffers[4]);
	NodeInfo* outInfos = (NodeInfo*)(buffers[5]);
	void* outGPs = (void*)(buffers[6]);
#ifdef TEST
	Timer t;
#endif
	switch (d.type)
	{
	case ElementType::BF16:
		crossover_kernel<nv_bfloat16>(stream, d, nodeInfos, prefixGPs, outInfos, outGPs, leftPerms, rightPerms, lrNodes);
		break;
	case ElementType::F16:
		crossover_kernel<half>(stream, d, nodeInfos, prefixGPs, outInfos, outGPs, leftPerms, rightPerms, lrNodes);
		break;
	case ElementType::F32:
		crossover_kernel<float>(stream, d, nodeInfos, prefixGPs, outInfos, outGPs, leftPerms, rightPerms, lrNodes);
		break;
	case ElementType::F64:
		crossover_kernel<double>(stream, d, nodeInfos, prefixGPs, outInfos, outGPs, leftPerms, rightPerms, lrNodes);
		break;
	default:
		throw std::runtime_error("Unsupported data type");
		break;
	}
#ifdef TEST
	auto err = cudaDeviceSynchronize();
	if (err != 0)
		throw std::runtime_error("Execution error of code " + (int)err);
	std::cout << t.elapsed() << std::endl;
#endif
}


template<typename T>
__global__ void treeGPMutationKernel(const NodeInfo* nodeInfos, const T* prefixGPs, const int* mutateIndices, const NodeInfo* newInfos, const T* newGPs, NodeInfo* outInfos, T* outGPs, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int maxNewGPLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	T* gp = (T*)alloca(MAX_STACK * sizeof(T));
	NodeInfo* info = (NodeInfo*)alloca(MAX_STACK * sizeof(NodeInfo));
	auto d_leftInfo = nodeInfos + n * maxGPLen;
	auto d_leftValue = prefixGPs + n * maxGPLen;
	auto d_rightInfo = newInfos + n * maxNewGPLen;
	auto d_rightValue = newGPs + n * maxNewGPLen;
	const unsigned int leftNode = mutateIndices[n];
	const unsigned int oldSubtreeSize = d_leftInfo[leftNode].subtreeSize, newSubtreeSize = d_rightInfo[0].subtreeSize;
	const unsigned int sizeDiff = newSubtreeSize - oldSubtreeSize, remainSize = d_leftInfo[0].subtreeSize - (leftNode + oldSubtreeSize);
	const unsigned int oldOffset = leftNode + oldSubtreeSize, newOffset = leftNode + newSubtreeSize;
	// copy
	for (int i = 0; i < leftNode; i++)
	{
		gp[i] = d_leftValue[i];
		info[i] = d_leftInfo[i];
	}
	for (int i = 0; i < newSubtreeSize; i++)
	{
		gp[i + leftNode] = d_rightValue[i];
		info[i + leftNode] = d_rightInfo[i];
	}
	for (int i = 0; i < remainSize; i++)
	{
		gp[i + newOffset] = d_leftValue[i + oldOffset];
		info[i + newOffset] = d_leftInfo[i + oldOffset];
	}
	// change subtree sizes of ancestors
	unsigned int current = 0;
	while (leftNode != current)
	{
		info[current].subtreeSize += sizeDiff;
		if (info[current + 1].subtreeSize + current + 1 > leftNode)
		{	// at left subtree
			current += 1;
		}
		else
		{	// at right subtree
			current += info[current + 1].subtreeSize + 1;
		}
	}
	// outupt
	const unsigned int len = info[0].subtreeSize;
	for (int i = 0; i < len; i++)
	{
		outGPs[i] = gp[i];
		outInfos[i] = info[i];
	}
}

template<typename T>
inline void mutation_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const NodeInfo* nodeInfos, const void* prefixGPs, const int* mutateIndices, const NodeInfo* newInfos, const void* newGPs, NodeInfo* outInfos, void* outGPs)
{
	treeGPMutationKernel<T><<<(d.popSize - 1) / 128 + 1, 128, 0, stream>>>(nodeInfos, (const T*)prefixGPs, mutateIndices, newInfos, (const T*)newGPs, outInfos, (T*)outGPs, d.popSize, d.gpLen, d.varLen);
}

void treeGP_mutation(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	// buffers: NodeInfo* nodeInfos, T* prefixGPs, int* mutateIndices, NodeInfo* newInfos, T* newGPs
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	NodeInfo* nodeInfos = (NodeInfo*)(buffers[0]);
	void* prefixGPs = (void*)(buffers[1]);
	const int* mutateIndices = (const int*)(buffers[2]);
	const NodeInfo* newInfos = (const NodeInfo*)(buffers[3]);
	const void* newGPs = (const void*)(buffers[4]);
	NodeInfo* outInfos = (NodeInfo*)(buffers[5]);
	void* outGPs = (void*)(buffers[6]);
#ifdef TEST
	Timer t;
#endif
	switch (d.type)
	{
	case ElementType::BF16:
		mutation_kernel<nv_bfloat16>(stream, d, nodeInfos, prefixGPs, mutateIndices, newInfos, newGPs, outInfos, outGPs);
		break;
	case ElementType::F16:
		mutation_kernel<half>(stream, d, nodeInfos, prefixGPs, mutateIndices, newInfos, newGPs, outInfos, outGPs);
		break;
	case ElementType::F32:
		mutation_kernel<float>(stream, d, nodeInfos, prefixGPs, mutateIndices, newInfos, newGPs, outInfos, outGPs);
		break;
	case ElementType::F64:
		mutation_kernel<double>(stream, d, nodeInfos, prefixGPs, mutateIndices, newInfos, newGPs, outInfos, outGPs);
		break;
	default:
		throw std::runtime_error("Unsupported data type");
		break;
	}
#ifdef TEST
	auto err = cudaDeviceSynchronize();
	if (err != 0)
		throw std::runtime_error("Execution error of code " + (int)err);
	std::cout << t.elapsed() << std::endl;
#endif
}