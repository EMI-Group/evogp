#include "kernel.h"
#include <malloc.h>

#ifdef _MSC_VER
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
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
#endif


template<typename T>
__global__ void treeGPEvalKernel(const GPNode<T>* gps, const T* variables, T* results, const unsigned int popSize, const unsigned int maxGPLen, const unsigned int varLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	T* stack = (T*)alloca(MAX_STACK * sizeof(T));
	unsigned int* infos = (unsigned int*)alloca(MAX_STACK * sizeof(unsigned int));
	auto i_gps = gps + n * maxGPLen;
	auto i_vars = variables + n * varLen;
	auto s_vars = stack + MAX_STACK - varLen;
	unsigned int len = i_gps[0].subtreeSize;
	for (int i = 0; i < varLen; i++)
	{
		s_vars[i] = i_vars[i];
	}
	for (int i = 0; i < len; i++)
	{
		stack[len - i - 1] = i_gps[i].value;
		infos[len - i - 1] = i_gps[i].nodeType;
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
inline void eval_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const void* gps, const void* variables, void* results)
{
	int gridSize{}, blockSize{};
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPEvalKernel<T>);
	if (gridSize * blockSize < d.popSize)
		gridSize = (d.popSize - 1) / blockSize + 1;
	treeGPEvalKernel<T><<<gridSize, blockSize, 0, stream>>>((const GPNode<T>*)gps, (const T*)variables, (T*)results, d.popSize, d.gpLen, d.varLen);
}

void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	const TreeGPDescriptor& d = *UnpackDescriptor<TreeGPDescriptor>(opaque, opaque_len);
	const void* gps = (const void*)(buffers[0]);
	const void* variables = (const void*)(buffers[1]);
	void* results = (void*)(buffers[2]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1000; i++)
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
__global__ void treeGPCrossoverKernel(const GPNode<T>* gps, const int* leftPerms, const int* rightPerms, const UInt16_2* lrNodes, GPNode<T>* outGPs, const unsigned int popSize, const unsigned int maxGPLen)
{
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= popSize)
		return;
	// init
	GPNode<T>* gp = (GPNode<T>*)alloca(MAX_STACK * sizeof(GPNode<T>));
	auto i_leftGP = gps + leftPerms[n] * maxGPLen;
	auto i_rightGP = gps + rightPerms[n] * maxGPLen;
	const int leftNode = lrNodes[n].left, rightNode = lrNodes[n].right;
	const int oldSubtreeSize = i_leftGP[leftNode].subtreeSize, newSubtreeSize = i_rightGP[rightNode].subtreeSize;
	const int sizeDiff = newSubtreeSize - oldSubtreeSize;
	const int oldOffset = leftNode + oldSubtreeSize;
	const int leftSize = i_leftGP[0].subtreeSize;
	// copy
	for (int i = 0; i < leftNode; i++)
	{
		gp[i] = i_leftGP[i];
	}
	for (int i = 0; i < newSubtreeSize; i++)
	{
		gp[i + leftNode] = i_rightGP[i + rightNode];
	}
	for (int i = oldOffset; i < leftSize; i++)
	{
		gp[i + sizeDiff] = i_leftGP[i];
	}
	// change subtree sizes of ancestors
	int current = 0;
	while (leftNode > current)
	{
		gp[current].subtreeSize += sizeDiff;
		auto rightTreeIndex = gp[current + 1].subtreeSize + current + 1;
		if (rightTreeIndex > leftNode)
		{	// at left subtree
			current += 1;
		}
		else
		{	// at right subtree
			current = rightTreeIndex;
		}
	}
	// outupt
	const int len = gp[0].subtreeSize;
	auto o_gp = outGPs + n * maxGPLen;
	for (int i = 0; i < len; i++)
	{
		o_gp[i] = gp[i];
	}
}

template<typename T>
inline void crossover_kernel(cudaStream_t stream, const TreeGPDescriptor& d, const void* gps, const int* leftPerms, const int* rightPerms, const UInt16_2* lrNodes, void* outGPs)
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
	const UInt16_2* lrNodes = (const UInt16_2*)(buffers[3]);
	void* outGPs = (void*)(buffers[4]);
#ifdef TEST
	Timer t;
	for (int i = 0; i < 1000; i++)
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
	GPNode<T>* gp = (GPNode<T>*)alloca(MAX_STACK * sizeof(GPNode<T>));
	auto i_leftGP = gps + n * maxGPLen;
	auto i_rightGP = newGPs + n * maxNewGPLen;
	const int leftNode = mutateIndices[n];
	const int oldSubtreeSize = i_leftGP[leftNode].subtreeSize, newSubtreeSize = i_rightGP[0].subtreeSize;
	const int sizeDiff = newSubtreeSize - oldSubtreeSize;
	const int oldOffset = leftNode + oldSubtreeSize;
	const int leftSize = i_leftGP[0].subtreeSize;
	// copy
	for (int i = 0; i < leftNode; i++)
	{
		gp[i] = i_leftGP[i];
	}
	for (int i = 0; i < newSubtreeSize; i++)
	{
		gp[i + leftNode] = i_rightGP[i];
	}
	for (int i = oldOffset; i < leftSize; i++)
	{
		gp[i + sizeDiff] = i_leftGP[i];
	}
	// change subtree sizes of ancestors
	int current = 0;
	while (leftNode > current)
	{
		gp[current].subtreeSize += sizeDiff;
		auto rightTreeIndex = gp[current + 1].subtreeSize + current + 1;
		if (rightTreeIndex > leftNode)
		{	// at left subtree
			current += 1;
		}
		else
		{	// at right subtree
			current = rightTreeIndex;
		}
	}
	// outupt
	const int len = gp[0].subtreeSize;
	auto o_gp = outGPs + n * maxGPLen;
	for (int i = 0; i < len; i++)
	{
		o_gp[i] = gp[i];
	}
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
	for (int i = 0; i < 1000; i++)
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


#ifdef _MSC_VER
int main()
{
	constexpr size_t popSize = 200000, maxGPLen = 16, gpSize = 16;
	GPNode<float> gp[gpSize]
	{
		GPNode<float>{Function::ADD, (uint16_t)NodeType::BFUNC, 16},
		GPNode<float>{Function::MUL, (uint16_t)NodeType::BFUNC, 5},
		GPNode<float>{Function::MUL, (uint16_t)NodeType::BFUNC, 3},
		GPNode<float>{0, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{0, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{0, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{Function::ADD, (uint16_t)NodeType::BFUNC, 10},
		GPNode<float>{Function::MUL, (uint16_t)NodeType::BFUNC, 3},
		GPNode<float>{1, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{1, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{Function::ADD, (uint16_t)NodeType::BFUNC, 6},
		GPNode<float>{Function::MUL, (uint16_t)NodeType::BFUNC, 3},
		GPNode<float>{0, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{1, (uint16_t)NodeType::VAR, 1},
		GPNode<float>{Function::SIN, (uint16_t)NodeType::UFUNC, 2},
		GPNode<float>{4, (uint16_t)NodeType::CONST, 1},
	};
	GPNode<float>* gps = new GPNode<float>[popSize * gpSize];
	UInt16_2* lrNodes = new UInt16_2[popSize];
	int* perms = new int[popSize];
	float* vars = new float[popSize * 2];
	for (size_t i = 0; i < popSize; i++)
	{
		memcpy(gps + i * gpSize, &gp, sizeof(gp));
		lrNodes[i] = UInt16_2{ (uint16_t)11, (uint16_t)11 };
		perms[i] = i;
		vars[i * 2] = 1; vars[i * 2 + 1] = 2;
	}
	GPNode<float>* d_gps, *o_gps;
	float *d_vars;
	UInt16_2* d_lrs;
	int* d_perms1, *d_perms2;
	cudaMalloc(&d_gps, popSize * maxGPLen * sizeof(GPNode<float>));
	cudaMalloc(&o_gps, popSize * maxGPLen * sizeof(GPNode<float>));
	cudaMalloc(&d_lrs, popSize * sizeof(UInt16_2));
	cudaMalloc(&d_perms1, popSize * sizeof(int));
	cudaMalloc(&d_perms2, popSize * sizeof(int));
	cudaMalloc(&d_vars, popSize * 2 * sizeof(float));
	cudaMemcpy2D(d_gps, maxGPLen * sizeof(GPNode<float>), gps, gpSize * sizeof(GPNode<float>), gpSize * sizeof(GPNode<float>), popSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_lrs, lrNodes, popSize * sizeof(UInt16_2), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_perms1, perms, popSize * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_perms2, perms, popSize * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_vars, vars, popSize * 2 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	thrust::default_random_engine g1(10);
	thrust::shuffle(thrust::cuda::par, d_perms1, d_perms1 + popSize, g1);
	thrust::default_random_engine g2(100);
	thrust::shuffle(thrust::cuda::par, d_perms2, d_perms2 + popSize, g2);

	TreeGPDescriptor d{ popSize, maxGPLen, 2, ElementType::F32 };
	eval_kernel<float>(NULL, d, d_gps, d_vars, o_gps);
	crossover_kernel<float>(NULL, d, d_gps, d_perms1, d_perms2, d_lrs, o_gps);
	cudaDeviceSynchronize();
}
#endif