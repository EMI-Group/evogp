#include "kernel.h"

#ifdef _MSC_VER
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#endif // _MSC_VER

//#ifdef TEST
#include <chrono> // for std::chrono functions

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_type = std::chrono::steady_clock;
	using seconi_type = std::chrono::duration<double, std::ratio<1> >;

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
		return std::chrono::duration_cast<seconi_type>(clock_type::now() - m_beg).count();
	}
};
//#endif // TEST


template<typename T, bool multiOutput = false>
__device__ inline void _treeGPEvalByStack(const GPNode<T>* i_gps, const T* i_vars, T* s_vals, uint16_t* s_infos, const unsigned int n, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int varLen, const unsigned int outLen, T*& s_outs, int& top)
{
	T* s_vars = (T*)(s_infos + MAX_STACK);
	if constexpr (multiOutput)
	{
		s_outs = (T*)(s_infos + MAX_STACK + MAX_STACK / 2);
	}
	const unsigned int len = i_gps[0].subtreeSize;
	for (int i = 0; i < varLen; i++)
	{
		s_vars[i] = i_vars[i];
	}
	for (int i = 0; i < outLen; i++)
	{
		s_outs[i] = 0;
	}
	for (int i = 0; i < len; i++)
	{
		s_vals[len - i - 1] = i_gps[i].value;
		s_infos[len - i - 1] = i_gps[i].nodeType;
	}
	// do stack operation according to the type of each node
	top = 0;
	for (int i = 0; i < len; i++)
	{
		uint16_t node_type = s_infos[i];
		uint16_t outNode = node_type & (uint16_t)NodeType::OUT_NODE;
		node_type &= NodeType::TYPE_MASK;
		T node_value = s_vals[i];

		if (node_type == NodeType::CONST)
		{
			s_vals[top++] = node_value;
			continue;
		}
		else if (node_type == NodeType::VAR)
		{
			int var_num = (int)node_value;
			s_vals[top++] = s_vars[var_num];
			continue;
		}
		unsigned int function, outIdx;
		function = (unsigned int)node_value;
		if constexpr (multiOutput)
		{
			if (outNode)
			{
				OutNodeValue<T> v = *(OutNodeValue<T>*) & node_value;
				function = v.function;
				outIdx = v.outIndex;
			}
		}
		T right_node{};
		T top_val{};
		if (node_type == NodeType::UFUNC)
		{
			T var1 = s_vals[--top];
			right_node = var1;
			if (function == Function::SIN)
			{
				top_val = std::sin(var1);
			}
			else if (function == Function::COS)
			{
				top_val = std::cos(var1);
			}
			else if (function == Function::SINH)
			{
				top_val = std::sinh(var1);
			}
			else if (function == Function::COSH)
			{
				top_val = std::cosh(var1);
			}
			else if (function == Function::LOG)
			{
				if (var1 == T(0.0f))
				{
					top_val = T(-MAX_VAL);
				}
				else
				{
					top_val = std::log(std::abs(var1));
				}
			}
			else if (function == Function::INV)
			{
				if (std::abs(var1) <= T(DELTA))
				{
					var1 = copy_sign(T(DELTA), var1);
				}
				top_val = T(1.0f) / var1;
			}
			else if (function == Function::EXP)
			{
				top_val = std::exp(var1);
			}
			else if (function == Function::NEG)
			{
				top_val = -var1;
			}
			else if (function == Function::ABS)
			{
				top_val = std::abs(var1);
			}
			else if (function == Function::SQRT)
			{
				if (var1 <= T(0.0f))
				{
					var1 = std::abs(var1);
				}
				top_val = std::sqrt(var1);
			}
		}
		else if (node_type == NodeType::BFUNC)
		{
			T var1 = s_vals[--top];
			T var2 = s_vals[--top];
			right_node = var2;
			if (function == Function::ADD)
			{
				top_val = var1 + var2;
			}
			else if (function == Function::SUB)
			{
				top_val = var1 - var2;
			}
			else if (function == Function::MUL)
			{
				top_val = var1 * var2;
			}
			else if (function == Function::DIV)
			{
				if (std::abs(var2) <= T(DELTA))
				{
					var2 = copy_sign(T(DELTA), var2);
				}
				top_val = var1 / var2;
			}
			else if (function == Function::POW)
			{
				if (var1 == T(0.0f) && var2 == T(0.0f))
				{
					top_val = T(0.0f);
				}
				else
				{
					top_val = std::pow(std::abs(var1), var2);
				}
			}
			else if (function == Function::MAX)
			{
				top_val = var1 >= var2 ? var1 : var2;
			}
			else if (function == Function::MIN)
			{
				top_val = var1 <= var2 ? var1 : var2;
			}
			else if (function == Function::LT)
			{
				top_val = var1 < var2 ? T(1) : T(-1);
			}
			else if (function == Function::GT)
			{
				top_val = var1 > var2 ? T(1) : T(-1);
			}
			else if (function == Function::LE)
			{
				top_val = var1 <= var2 ? T(1) : T(-1);
			}
			else if (function == Function::GE)
			{
				top_val = var1 >= var2 ? T(1) : T(-1);
			}
		}
		else //// if (node_type == NodeType::TFUNC)
		{
			T var1 = s_vals[--top];
			T var2 = s_vals[--top];
			T var3 = s_vals[--top];
			right_node = var3;
			//// if (function == Function::IF)
			top_val = var1 > T(0.0f) ? var2 : var3;
		}
		// multiple output
		if constexpr (multiOutput)
		{
			top_val = right_node;
			if (outNode && outIdx < outLen)
				s_outs[outIdx] += top_val;
		}
		// clip value
		if (is_nan(top_val))
		{
			s_vals[top] = T(0);
		}
		else if (is_inf(top_val) || std::abs(top_val) > T(MAX_VAL))
		{
			s_vals[top] = copy_sign(T(MAX_VAL), top_val);
		}
		else
		{
			s_vals[top] = top_val;
		}
		top++;
	}
	// return
}


template<typename T, bool multiOutput = false>
__global__ void treeGPEvalKernel(const GPNode<T>* gps, const T* variables, T* results, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int varLen, const unsigned int outLen = 0)
{
	const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	if constexpr (multiOutput)
	{
		assert(outLen > 0);
		assert(varLen * sizeof(T) / sizeof(int) <= MAX_STACK / 4);
		assert(outLen * sizeof(T) / sizeof(int) <= MAX_STACK / 4);
	}
	else
	{
		assert(varLen * sizeof(T) / sizeof(int) <= MAX_STACK / 2);
	}
	// init
	T* stack = (T*)alloca(MAX_STACK * sizeof(T));
	uint16_t* infos = (uint16_t*)alloca(MAX_STACK * sizeof(int));
	auto i_gps = gps + n * maxGPLen;
	auto i_vars = variables + n * varLen;
	// call
	T* s_outs{};
	int top{};
	_treeGPEvalByStack<T, multiOutput>(i_gps, i_vars, stack, infos, n, popSize, maxGPLen, varLen, outLen, s_outs, top);
	// final
	if constexpr (multiOutput)
	{
		auto o_res = results + n * outLen;
		for (int i = 0; i < outLen; i++)
		{
			o_res[i] = s_outs[i];
		}
	}
	else
	{
		results[n] = stack[--top];
	}
}

template<typename T>
inline void eval_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const void* gps, const void* variables, void* results)
{
	int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPEvalKernel<T>);
	if (gridSize * blockSize < d.popSize)
		gridSize = (d.popSize - 1) / blockSize + 1;
	if (d.outLen > 1)
		treeGPEvalKernel<T, true><<<gridSize, blockSize, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (T*)results, d.popSize, d.gpLen, d.varLen, d.outLen);
	else
		treeGPEvalKernel<T, false><<<gridSize, blockSize, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (T*)results, d.popSize, d.gpLen, d.varLen, 0);
}

void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const void* gps = (const void*)(buffers[0]);
	const void* variables = (const void*)(buffers[1]);
	void* results = (void*)(buffers[2]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			eval_kernel<float>(stream, d, gps, variables, results);
			break;
		case ElementType::F64:
			eval_kernel<double>(stream, d, gps, variables, results);
			break;
		default:
			throw std::runtime_error("Unsupported data type");
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}

template<typename T>
__host__ __device__
inline void _gpTreeReplace(const int leftNode, const int rightNode, const int newSubtreeSize, const int oldOffset, const int leftSize, const int sizeDiff, const GPNode<T>* i_leftGP, const GPNode<T>* i_rightGP, GPNode<T>* o_gp)
{
	// copy until replace position
	GPNode<T>* gp = (GPNode<T>*)alloca(MAX_STACK * sizeof(GPNode<T>));
	for (int i = 0; i < leftNode; i++)
	{
		gp[i] = i_leftGP[i];
	}
	// change subtree sizes of ancestors
	int current = 0;
	while (leftNode > current)
	{
		int midTreeIndex{}, rightTreeIndex{};
		gp[current].subtreeSize += sizeDiff;
		auto node_type = gp[current].nodeType;
		node_type &= NodeType::TYPE_MASK;
		current++;
		switch (node_type)
		{
		case NodeType::UFUNC:
			// do nothing
			break;
		case NodeType::BFUNC:
			rightTreeIndex = gp[current].subtreeSize + current;
			if (rightTreeIndex > leftNode)
			{	// at left subtree
				// do nothing
			}
			else
			{	// at right subtree
				current = rightTreeIndex;
			}
			break;
		case NodeType::TFUNC:
			midTreeIndex = gp[current].subtreeSize + current;
			if (midTreeIndex > leftNode)
			{	// at left subtree
				// do nothing
				break;
			}
			rightTreeIndex = gp[midTreeIndex].subtreeSize + midTreeIndex;
			if (rightTreeIndex > leftNode)
			{	// at mid subtree
				current = midTreeIndex;
			}
			else
			{	// at right subtree
				current = rightTreeIndex;
			}
			break;
		default:
			break;
		}
	}
	// copy rest
	for (int i = 0; i < newSubtreeSize; i++)
	{
		gp[i + leftNode] = i_rightGP[i + rightNode];
	}
	for (int i = oldOffset; i < leftSize; i++)
	{
		gp[i + sizeDiff] = i_leftGP[i];
	}
	// output
	const int len = gp[0].subtreeSize;
	for (int i = 0; i < len; i++)
	{
		o_gp[i] = gp[i];
	}
}


template<typename T>
__global__ void treeGPCrossoverKernel(const GPNode<T>* gps, const int* leftPerms, const int* rightPerms, const LeftRightIdx* lrNodes, GPNode<T>* outGPs, const unsigned int popSize, const unsigned int maxGPLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	auto o_gp = outGPs + n * maxGPLen;
	auto i_leftGP = gps + leftPerms[n] * maxGPLen;
	const int leftSize = i_leftGP[0].subtreeSize;
	if (rightPerms[n] < 0 || rightPerms[n] >= popSize)
	{	// invalid right
		for (int i = 0; i < leftSize; i++)
		{
			o_gp[i] = i_leftGP[i];
		}
		return;
	}
	auto i_rightGP = gps + rightPerms[n] * maxGPLen;
	const int leftNode = lrNodes[n].left, rightNode = lrNodes[n].right;
	const int oldSubtreeSize = i_leftGP[leftNode].subtreeSize, newSubtreeSize = i_rightGP[rightNode].subtreeSize;
	const int sizeDiff = newSubtreeSize - oldSubtreeSize;
	const int oldOffset = leftNode + oldSubtreeSize;
	if (leftSize + sizeDiff > maxGPLen)
	{	// too large output size
		for (int i = 0; i < leftSize; i++)
		{
			o_gp[i] = i_leftGP[i];
		}
		return;
	}
	// replace
	_gpTreeReplace(leftNode, rightNode, newSubtreeSize, oldOffset, leftSize, sizeDiff, i_leftGP, i_rightGP, o_gp);
}

template<typename T>
inline void crossover_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const void* gps, const int* leftPerms, const int* rightPerms, const LeftRightIdx* lrNodes, void* outGPs)
{
	int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPCrossoverKernel<T>);
	if (gridSize * blockSize < d.popSize)
		gridSize = (d.popSize - 1) / blockSize + 1;
	treeGPCrossoverKernel<T><<<gridSize, blockSize, 0, stream>>>((const GPNode<T>*)gps, leftPerms, rightPerms, lrNodes, (GPNode<T>*)outGPs, d.popSize, d.gpLen);
}

void treeGP_crossover(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const void* gps = (const void*)(buffers[0]);
	const int* leftPerms = (const int*)(buffers[1]);
	const int* rightPerms = (const int*)(buffers[2]);
	const LeftRightIdx* lrNodes = (const LeftRightIdx*)(buffers[3]);
	void* outGPs = (void*)(buffers[4]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			crossover_kernel<float>(stream, d, gps, leftPerms, rightPerms, lrNodes, outGPs);
			break;
		case ElementType::F64:
			crossover_kernel<double>(stream, d, gps, leftPerms, rightPerms, lrNodes, outGPs);
			break;
		default:
			throw std::runtime_error("Unsupported data type");
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}


template<typename T>
__global__ void treeGPMutationKernel(const GPNode<T>* gps, const int* mutateIndices, const GPNode<T>* newGPs, GPNode<T>* outGPs, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int maxNewGPLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	auto o_gp = outGPs + n * maxGPLen;
	auto i_leftGP = gps + n * maxGPLen;
	const int leftNode = mutateIndices[n];
	const int leftSize = i_leftGP[0].subtreeSize;
	if (leftNode < 0 || leftNode >= leftSize)
	{	// invalid node index
		for (int i = 0; i < leftSize; i++)
		{
			o_gp[i] = i_leftGP[i];
		}
		return;
	}
	auto i_rightGP = newGPs + n * maxNewGPLen;
	const int oldSubtreeSize = i_leftGP[leftNode].subtreeSize, newSubtreeSize = i_rightGP[0].subtreeSize;
	const int sizeDiff = newSubtreeSize - oldSubtreeSize;
	const int oldOffset = leftNode + oldSubtreeSize;
	if (leftSize + sizeDiff > maxGPLen)
	{	// too large output size
		for (int i = 0; i < leftSize; i++)
		{
			o_gp[i] = i_leftGP[i];
		}
		return;
	}
	// replace
	_gpTreeReplace(leftNode, 0, newSubtreeSize, oldOffset, leftSize, sizeDiff, i_leftGP, i_rightGP, o_gp);
}

template<typename T>
inline void mutation_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const void* gps, const int* mutateIndices, const void* newGPs, void* outGPs)
{
	int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPMutationKernel<T>);
	if (gridSize * blockSize < d.popSize)
		gridSize = (d.popSize - 1) / blockSize + 1;
	treeGPMutationKernel<T><<<gridSize, blockSize, 0, stream>>>((const GPNode<T>*)gps, mutateIndices, (const GPNode<T>*)newGPs, (GPNode<T>*)outGPs, d.popSize, d.gpLen, d.varLen);
}

void treeGP_mutation(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const void* gps = (const void*)(buffers[0]);
	const int* mutateIndices = (const int*)(buffers[1]);
	const void* newGPs = (const void*)(buffers[2]);
	void* outGPs = (void*)(buffers[3]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			mutation_kernel<float>(stream, d, gps, mutateIndices, newGPs, outGPs);
			break;
		case ElementType::F64:
			mutation_kernel<double>(stream, d, gps, mutateIndices, newGPs, outGPs);
			break;
		default:
			throw std::runtime_error("Unsupported data type");
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}


constexpr auto SR_BLOCK_SIZE = 1024;

template<typename T, bool multiOutput = false, bool useMSE = true>
__global__ void treeGPRegressionFitnessKernel(const GPNode<T>* gps, const T* variables, const T* labels, T* fitnesses, const unsigned int popSize, const unsigned int dataPoints, const unsigned int maxGPLen, const unsigned int varLen, const unsigned int outLen = 0)
{
	const unsigned int maxThreadBlocks = (dataPoints - 1) / SR_BLOCK_SIZE + 1;
	const unsigned int nGP = blockIdx.x, nTB = blockIdx.y, threadId = threadIdx.x;
	const unsigned int dataPointId = nTB * SR_BLOCK_SIZE + threadId;
	if (nGP >= popSize || nTB >= maxThreadBlocks)
		return;
	if constexpr (multiOutput)
	{
		assert(outLen > 0);
		assert(varLen * sizeof(T) / sizeof(int) <= MAX_STACK / 4);
		assert(outLen * sizeof(T) / sizeof(int) <= MAX_STACK / 4);
	}
	else
	{
		assert(varLen * sizeof(T) / sizeof(int) <= MAX_STACK / 2);
	}
	// init
	__shared__ T sharedFitness[SR_BLOCK_SIZE];
	T fit = T(0.0f);
	T* stack = (T*)alloca(MAX_STACK * sizeof(T));
	uint16_t* infos = (uint16_t*)alloca(MAX_STACK * sizeof(int));
	auto i_gps = gps + nGP * maxGPLen;
	// evaluate over data points
	if (dataPointId < dataPoints)
	{
		// eval
		auto i_vars = variables + dataPointId * varLen;
		T* s_outs{};
		int top{};
		_treeGPEvalByStack<T, multiOutput>(i_gps, i_vars, stack, infos, nGP, popSize, maxGPLen, varLen, outLen, s_outs, top);
		// accumulate
		if constexpr (multiOutput)
		{
			auto i_labels = labels + dataPointId * outLen;
			for (int i = 0; i < outLen; i++)
			{
				T diff = i_labels[i] - s_outs[i];
				if constexpr (useMSE)
					fit += diff * diff;
				else
					fit += std::abs(diff);
			}
		}
		else
		{
			T diff = labels[dataPointId] - stack[--top];
			if constexpr (useMSE)
				fit = diff * diff;
			else
				fit = std::abs(diff);
		}
	}
	// write to shared memory
	sharedFitness[threadId] = fit;
	__syncthreads();
	// reduce
#define __REDUCE_SHARED(size) if (threadId < size) { sharedFitness[threadId] = T(0.5f) * (sharedFitness[threadId] + sharedFitness[threadId + size]); } __syncthreads()
	if constexpr (SR_BLOCK_SIZE >= 1024)
	{
		__REDUCE_SHARED(512);
	}
	if constexpr (SR_BLOCK_SIZE >= 512)
	{
		__REDUCE_SHARED(256);
	}
	if constexpr (SR_BLOCK_SIZE >= 256)
	{
		__REDUCE_SHARED(128);
	}
	__REDUCE_SHARED(64);
	__REDUCE_SHARED(32);
	__REDUCE_SHARED(16);
	__REDUCE_SHARED(8);
	__REDUCE_SHARED(4);
	if (threadId == 0)
	{
		T finalFit = T(0.25f) * (sharedFitness[threadId] + sharedFitness[threadId + 1] + sharedFitness[threadId + 2] + sharedFitness[threadId + 3]);
		atomicAdd(fitnesses + nGP, finalFit / maxThreadBlocks);
	}
#undef __REDUCE_SHARED
}

template<typename T>
inline void SR_fitness_kernel(cudaStream_t stream, const TreeGPSRDescriptor& d, const void* gps, const void* variables, const void* labels, void* fitnesses)
{
	const unsigned int threadBlocks = (d.dataPoints - 1) / SR_BLOCK_SIZE + 1;
	dim3 gridSize{ (unsigned int)d.popSize, threadBlocks };
	auto err = cudaMemsetAsync(fitnesses, 0, d.popSize * sizeof(T), stream);
	if (d.outLen > 1)
	{
		if (d.useMSE)
			treeGPRegressionFitnessKernel<T, true, true><<<gridSize, SR_BLOCK_SIZE, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (const T*)labels, (T*)fitnesses, d.popSize, d.dataPoints, d.gpLen, d.varLen, d.outLen);
		else
			treeGPRegressionFitnessKernel<T, true, false><<<gridSize, SR_BLOCK_SIZE, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (const T*)labels, (T*)fitnesses, d.popSize, d.dataPoints, d.gpLen, d.varLen, d.outLen);
	}
	else
	{
		if (d.useMSE)
			treeGPRegressionFitnessKernel<T, false, true><<<gridSize, SR_BLOCK_SIZE, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (const T*)labels, (T*)fitnesses, d.popSize, d.dataPoints, d.gpLen, d.varLen, 0);
		else
			treeGPRegressionFitnessKernel<T, false, false><<<gridSize, SR_BLOCK_SIZE, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (const T*)labels, (T*)fitnesses, d.popSize, d.dataPoints, d.gpLen, d.varLen, 0);
	}
}

void treeGP_SR_fitness(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPSRDescriptor& d = *UnpackDescriptor<TreeGPSRDescriptor>(opaque, opaque_len);
	const void* gps = (const void*)(buffers[0]);
	const void* variables = (const void*)(buffers[1]);
	const void* labels = (const void*)(buffers[2]);
	void* fitnesses = (void*)(buffers[3]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			SR_fitness_kernel<float>(stream, d, gps, variables, labels, fitnesses);
			break;
		case ElementType::F64:
			SR_fitness_kernel<double>(stream, d, gps, variables, labels, fitnesses);
			break;
		default:
			throw std::runtime_error("Unsupported data type");
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}


template<typename T, typename RandomEngine, bool multiOutput = false>
__global__ void treeGPGenerate(GPNode<T>* results, const unsigned int* keys, const T* depth2leafProbs, const T* rouletteFuncs, const T* constSamples, const GPGenerateInfo info)
{
	const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= info.popSize)
		return;
	// possible registers init
	T leafProbs[MAX_FULL_DEPTH]{}, funcRoulette[Function::END]{};
	RandomEngine engine(hash(n, keys[0], keys[1]));
	thrust::uniform_real_distribution<T> rand(T(0.0f), T(1.0f));
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
	// stacks / local memory init
	GPNode<T>* gp = (GPNode<T>*)alloca(MAX_STACK * sizeof(GPNode<T>));
	NchildDepth* childsAndDepth = (NchildDepth*)alloca(MAX_STACK * sizeof(NchildDepth));
	childsAndDepth[0] = { 1, 0 };
	int topGP = 0, top = 1;
	// generate
	while (top > 0)
	{
		NchildDepth cd = childsAndDepth[--top];
		cd.childs--;
		NchildDepth cdNew{};
		GPNode<T> node{};
		if (rand(engine) >= leafProbs[cd.depth])
		{	// generate non-leaf (function) node
			T r = rand(engine);
			int k = 0;
#pragma unroll
			for (int i = Function::END - 1; i >= 0; i--)
			{
				if (r >= funcRoulette[i])
				{
					k = i + 1;
					break;
				}
			}
			typename GPNode<T>::U type = k <= Function::IF ? NodeType::TFUNC : k <= Function::GE ? NodeType::BFUNC : NodeType::UFUNC;
			if constexpr (multiOutput)
			{
				if (rand(engine) <= info.outProb)
				{	// output node
					typename GPNode<T>::U outType = type + NodeType::OUT_NODE;
					OutNodeValue<T> outNode{ (typename GPNode<T>::U)k, (typename GPNode<T>::U)(engine() % info.outLen) };
					node = GPNode<T>{ outNode, outType, 1 };
				}
			}
			// normal node
			node = GPNode<T>{ T(k), type, 1 };
			cdNew = NchildDepth{ uint16_t(type - 1), uint16_t(cd.depth + 1) };
		}
		else
		{	// generate leaf node
			T value{};
			typename GPNode<T>::U type{};
			if (rand(engine) <= info.constProb)
			{	// constant
				value = constSamples[engine() % info.constSamplesLen];
				type = NodeType::CONST;
			}
			else
			{	// variable
				value = engine() % info.varLen;
				type = NodeType::VAR;
			}
			node = GPNode<T>{ value, type, 1 };
		}
		gp[topGP++] = node;
		if (cd.childs > 0)
			childsAndDepth[top++] = cd;
		if (cdNew.childs > 0)
			childsAndDepth[top++] = cdNew;
	}
	// get subtree sizes
	int* nodeSize = (int*)childsAndDepth;
	top = 0;
	for (int i = topGP - 1; i >= 0; i--)
	{
		uint16_t node_type = gp[i].nodeType;
		node_type &= NodeType::TYPE_MASK;
		if (node_type <= NodeType::CONST)
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
		gp[i].subtreeSize = (typename GPNode<T>::U)nodeSize[top];
		top++;
	}
	const int len = gp[0].subtreeSize;
	auto o_gp = results + n * info.gpLen;
	for (int i = 0; i < len; i++)
	{
		o_gp[i] = gp[i];
	}
}

template<typename T>
inline void generate_kernel(cudaStream_t stream, const TreeGPGenerateDescriptor& d, const void* keys, void* gps, const void* depth2leafProbs, const void* rouletteFuncs, const void* constSamples)
{
#define __GEN(engine, multiout) do { \
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPGenerate<T, engine, multiout>); \
	if (gridSize * blockSize < d.popSize) \
	{ \
		gridSize = (d.popSize - 1) / blockSize + 1; \
	} \
	treeGPGenerate<T, engine, multiout><<<gridSize, blockSize, 0, stream>>>((GPNode<T>*)gps, (const unsigned int*)keys, (const T*)depth2leafProbs, (const T*)rouletteFuncs, (const T*)constSamples, d); \
} while (0)

	int gridSize{}, blockSize{};
	switch (d.engine)
	{
	case RandomEngine::Default:
		if (d.outLen > 1)
			__GEN(thrust::random::default_random_engine, true);
		else
			__GEN(thrust::random::default_random_engine, false);
		break;
	case RandomEngine::RANLUX24:
		if (d.outLen > 1)
			__GEN(thrust::random::ranlux24, true);
		else
			__GEN(thrust::random::ranlux24, false);
		break;
	case RandomEngine::RANLUX48:
		if (d.outLen > 1)
			__GEN(thrust::random::ranlux48, true);
		else
			__GEN(thrust::random::ranlux48, false);
		break;
	case RandomEngine::TAUS88:
		if (d.outLen > 1)
			__GEN(thrust::random::taus88, true);
		else
			__GEN(thrust::random::taus88, false);
		break;
	default:
		throw new std::runtime_error("Unsupported random engine.");
	}
#undef __GEN
}

void treeGP_generate(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPGenerateDescriptor& d = *UnpackDescriptor<TreeGPGenerateDescriptor>(opaque, opaque_len);
	// depth2leafProbs, const void* rouletteFuncs, const void* constSamples
	const void* keys = (const void*)(buffers[0]);
	const void* depth2leafProbs = (const void*)(buffers[1]);
	const void* rouletteFuncs = (const void*)(buffers[2]);
	const void* constSamples = (const void*)(buffers[3]);
	void* out_gps = (void*)(buffers[4]);

#ifdef TEST
	Timer t;
	for (int i = 0; i < 1; i++)
	{
#endif
		switch (d.type)
		{
		case ElementType::F32:
			generate_kernel<float>(stream, d, keys, out_gps, depth2leafProbs, rouletteFuncs, constSamples);
			break;
		case ElementType::F64:
			generate_kernel<double>(stream, d, keys, out_gps, depth2leafProbs, rouletteFuncs, constSamples);
			break;
		default:
			throw std::runtime_error("Unsupported data type");
			break;
		}
#ifdef TEST
	}
	auto err = cudaDeviceSynchronize();
	std::cout << "C++: " << t.elapsed() << std::endl;
	if (err != 0)
		std::cout << "Execution error of code " << (int)err << std::endl;
#endif
}