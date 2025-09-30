#pragma once

#include "defs.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/random.h>

#include <cmath>
#include <malloc.h>
#include <cassert>
#include <iostream>
#include <limits>

// enum RandomEngine {	Default, RANLUX24, RANLUX48, TAUS88 };

// #define RandomEngine thrust::random::default_random_engine
// using TAUS88;
#define RandomEngine thrust::random::taus88


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
);

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
);

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
);

void evaluate(
    const unsigned int popSize, 
    const unsigned int maxGPLen, 
    const unsigned int varLen, 
    const unsigned int outLen, 
    const float* value,
    const int16_t* type,
    const int16_t* subtree_size,
    const float* variables, 
    float* results
);

void SR_fitness(
	const unsigned int popSize,
	const unsigned int dataPoints,
	const unsigned int gpLen,
	const unsigned int varLen,
	const unsigned int outLen,
	const bool useMSE,
	const float* value,
	const int16_t* type,
	const int16_t* subtree_size,
	const float* variables, 
	const float* labels, 
	float* fitnesses,
	const unsigned int kernel_type = 0
);

struct GPNode
{
	float value;
	int16_t nodeType, subtreeSize;
};

struct OutNodeValue
{
	int16_t function, outIndex; // lower 16 bits: function, upper 16 bits: out index

	__host__ __device__ inline operator float() const
	{
		return *(float*)this;
	}
};

struct LeftRightIdx
{
	int16_t left;
	int16_t right;
};

struct NchildDepth
{
	int16_t childs, depth;
};


__host__ __device__
inline float copy_sign(const float number, const float sign)
{
#ifdef _MSC_VER
	return std::abs(number) * float(sign >= float(0) ? 1 : -1);
#else
	return std::copysign(number, sign);
#endif // _MSC_VER
}

__host__ __device__
inline bool is_inf(const float number)
{
#ifdef _MSC_VER
	return std::abs(number) == std::numeric_limits<float>::infinity();
#else
	return std::isinf(number);
#endif // _MSC_VER
}

__host__ __device__
inline bool is_nan(const float number)
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