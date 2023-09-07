#pragma once

#include "gpdefs.h"
#include "helpers.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>


#define HEADER __host__ __device__


enum ElementType { BF16, F16, F32, F64 };

struct TreeGPDescriptor
{
	unsigned int popSize, gpLen, varLen;
	ElementType type;
};

void treeGP_eval(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_crossover(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);

void treeGP_mutation(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);