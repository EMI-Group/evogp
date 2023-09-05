#include "kernel.h"

template<typename T>
struct TreeGPEval
{
	const uint8_t* nodeTypes;
	const T* prefixGPs;
	const unsigned int maxGPLen;
	const T* variables;
	const unsigned int varLen;

	TreeGPEval(const uint8_t* nodeTypes, const T* prefixGPs, const unsigned int maxGPLen, const T* variables, const unsigned int varLen)
		: nodeTypes(nodeTypes), prefixGPs(prefixGPs), maxGPLen(maxGPLen), variables(variables), varLen(varLen)
	{}


	HEADER T operator()(const unsigned int n, const int len)
	{
		// init
		T stack[MAX_STACK / sizeof(T)]{};
		int top = 0;
		auto d_nodeType = nodeTypes + n * maxGPLen;
		auto d_nodeValue = prefixGPs + n * maxGPLen;
		auto d_vars = variables + n * varLen;

		// do stack operation according to the type of each node
		for (int i = len - 1; i >= 0; i--) {
			int node_type = d_nodeType[i];
			T node_value = d_nodeValue[i];

			if (node_type == NodeType::CONST) {
				stack[top] = node_value;
				top++;
			}
			else if (node_type == NodeType::VAR) {
				int var_num = node_value;
				stack[top] = d_vars[var_num];
				top++;
			}
			else if (node_type == NodeType::UFUNC) {
				int function = node_value;
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
				int function = node_value;
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
		top--;
		return stack[top];
	}
};


template<typename T>
inline void eval(cudaStream_t stream, const TreeGPEvalDescriptor& d, const int* gpLengths, const uint8_t* nodeTypes, const void* prefixGPs, const void* variables, void* results)
{
	TreeGPEval<T> evaluator(nodeTypes, (const T*)prefixGPs, d.gpLen, (const T*)variables, d.varLen);
	auto c = thrust::make_counting_iterator((unsigned int)0);
	thrust::transform(thrust::cuda::par.on(stream), c, c + d.popSize, gpLengths, (T*)results, evaluator);
}

void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
{
	// buffers: int gpLengths[popSize], uint8_t nodeTypes[popSize, gpLen], T prefixGPs[popSize, gpLen], T variables[popSize, varLen], T results[popSize]
	const TreeGPEvalDescriptor& d = *UnpackDescriptor<TreeGPEvalDescriptor>(opaque, opaque_len);
	const int* gpLengths = (const int*)(buffers[0]);
	const uint8_t* nodeTypes = (const uint8_t*)(buffers[1]);
	const void* prefixGPs = (const void*)(buffers[2]);
	const void* variables = (const void*)(buffers[3]);
	void* results = (void*)(buffers[4]);
	switch (d.type)
	{
	case ElementType::BF16:
		eval<nv_bfloat16>(stream, d, gpLengths, nodeTypes, prefixGPs, variables, results);
		break;
	case ElementType::F16:
		eval<half>(stream, d, gpLengths, nodeTypes, prefixGPs, variables, results);
		break;
	case ElementType::F32:
		eval<float>(stream, d, gpLengths, nodeTypes, prefixGPs, variables, results);
		break;
	case ElementType::F64:
		eval<double>(stream, d, gpLengths, nodeTypes, prefixGPs, variables, results);
		break;
	default:
		throw std::runtime_error("Unsupported data type");
		break;
	}
}