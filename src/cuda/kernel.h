#pragma once

#include "gpdefs.h"
#include "helpers.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/random.h>

#include <cmath>
#include <malloc.h>
#include <cassert>
#include <iostream>
#include <limits>


#define HEADER __host__ __device__


enum ElementType { BF16, F16, F32, F64 };

enum RandomEngine {	Default, RANLUX24, RANLUX48, TAUS88 };


void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_crossover(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_mutation(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_SR_fitness(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_generate(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);


template<typename T>
struct GPNode
{
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double types are supported.");
	using U = std::conditional_t<std::is_same_v<T, float>, uint16_t, uint32_t>;
	T value;
	U nodeType, subtreeSize;
};

template<typename T>
struct OutNodeValue
{
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double types are supported.");
	using U = std::conditional_t<std::is_same_v<T, float>, uint16_t, uint32_t>;
	U function, outIndex;

	__host__ __device__ inline operator T() const
	{
		return *(T*)this;
	}
};

struct LeftRightIdx
{
	uint16_t left;
	uint16_t right;
};

struct NchildDepth
{
	uint16_t childs, depth;
};


struct alignas(float) TreeGPDescriptor
{
	int popSize, gpLen, varLen, outLen;
	ElementType type;

	TreeGPDescriptor(int popSize, int gpLen, int varLen, int outLen, const ElementType type)
		: popSize(popSize), gpLen(gpLen), varLen(varLen), outLen(outLen), type(type)
	{
	}
};

struct TreeGPSRDescriptor
{
	int popSize, dataPoints, gpLen, varLen, outLen;
	ElementType type;
	bool useMSE;

	TreeGPSRDescriptor(int popSize, int dataPoints, int gpLen, int varLen, int outLen, const ElementType type, bool useMSE)
		: popSize(popSize), dataPoints(dataPoints), gpLen(gpLen), varLen(varLen), outLen(outLen), type(type), useMSE(useMSE)
	{
	}
};

struct alignas(float) GPGenerateInfo
{
	unsigned int popSize, gpLen, varLen, outLen, constSamplesLen;
	float outProb, constProb;

	GPGenerateInfo(unsigned int popSize, unsigned int gpLen, unsigned int varLen, unsigned int outLen, unsigned int constSamplesLen, float outProb, float constProb)
		: popSize(popSize), gpLen(gpLen), varLen(varLen), outLen(outLen), constSamplesLen(constSamplesLen), outProb(outProb), constProb(constProb)
	{
	}
};

struct alignas(float) TreeGPGenerateDescriptor
{
	int popSize, gpLen, varLen, outLen, constSamplesLen;
	float outProb, constProb;
	RandomEngine engine;
	ElementType type;

	operator GPGenerateInfo() const
	{
		return GPGenerateInfo(popSize, gpLen, varLen, outLen, constSamplesLen, outProb, constProb);
	}

	TreeGPGenerateDescriptor(int popSize, int gpLen, int varLen, int outLen, int constSamplesLen, float outProb, float constProb, const RandomEngine engine, const ElementType type)
		: popSize(popSize), gpLen(gpLen), varLen(varLen), outLen(outLen), constSamplesLen(constSamplesLen), outProb(outProb), constProb(constProb), engine(engine), type(type)
	{
	}
};

template <typename T>
__host__ __device__
inline T copy_sign(const T number, const T sign)
{
#ifdef _MSC_VER
	return std::abs(number) * T(sign >= T(0) ? 1 : -1);
#else
	return std::copysign(number, sign);
#endif // _MSC_VER
}

template <typename T>
__host__ __device__
inline bool is_inf(const T number)
{
#ifdef _MSC_VER
	return std::abs(number) == std::numeric_limits<T>::infinity();
#else
	return std::isinf(number);
#endif // _MSC_VER
}

template <typename T>
__host__ __device__
inline bool is_nan(const T number)
{
#ifdef _MSC_VER
	return !(number == number);
#else
	return std::isnan(number);
#endif // _MSC_VER
}

constexpr size_t _FNV_offset_basis = 14695981039346656037ULL;
constexpr size_t _FNV_prime = 1099511628211ULL;

__host__ __device__
inline unsigned int hash(const unsigned int n, const unsigned int k1, const unsigned int k2)
{
	const unsigned int a[3]{ n, k1, k2 };
	auto h = _FNV_offset_basis;
	auto b = &reinterpret_cast<const unsigned char&>(a);
	constexpr auto C = sizeof(unsigned int) * 3;
	// accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
	for (size_t i = 0; i < C; ++i) {
		h ^= static_cast<size_t>(b[i]);
		h *= _FNV_prime;
	}
	return (unsigned int)h;
	//a = (a + 0x7ed55d16) + (a << 12);
	//a = (a ^ 0xc761c23c) ^ (a >> 19);
	//a = (a + 0x165667b1) + (a << 5);
	//a = (a + 0xd3a2646c) ^ (a << 9);
	//a = (a + 0xfd7046c5) + (a << 3);
	//a = (a ^ 0xb55a4f09) ^ (a >> 16);
	//return a;
}